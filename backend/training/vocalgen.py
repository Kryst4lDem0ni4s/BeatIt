import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import json
import pickle
import random
import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoFeatureExtractor
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def prepare_dataset(
    audio_dir, 
    output_dir, 
    split_ratio=0.9, 
    min_duration=1.0, 
    max_duration=30.0
):
    """Prepare dataset from raw audio files with lyrics"""
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    for directory in [
        train_dir, val_dir,
        os.path.join(train_dir, "audio"),
        os.path.join(train_dir, "references"),
        os.path.join(val_dir, "audio"),
        os.path.join(val_dir, "references")
    ]:
        os.makedirs(directory, exist_ok=True)
    
    # Find all audio files
    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith((".wav", ".mp3", ".flac")):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process each file
    train_samples = []
    val_samples = []
    
    for i, audio_path in enumerate(tqdm.tqdm(audio_files)):
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
            
            # Skip if duration is outside range
            if duration < min_duration or duration > max_duration:
                continue
            
            # Look for lyrics file
            lyrics_path = os.path.splitext(audio_path)[0] + ".txt"
            lyrics = ""
            if os.path.exists(lyrics_path):
                with open(lyrics_path, "r", encoding="utf-8") as f:
                    lyrics = f.read().strip()
            
            # Create sample metadata
            filename = f"{i:06d}.wav"
            sample = {
                "audio_file": filename,
                "original_path": audio_path,
                "duration": duration,
                "lyrics": lyrics,
                # Add default values for other attributes
                "gender": 0,  # neutral
                "accent": 0,
                "style": 0,
                "tempo": 120,
                "pitch": 0
            }
            
            # Decide if sample goes to train or validation set
            if random.random() < split_ratio:
                # Save audio file to train directory
                output_path = os.path.join(train_dir, "audio", filename)
                librosa.output.write_wav(output_path, audio, sr)
                train_samples.append(sample)
            else:
                # Save audio file to validation directory
                output_path = os.path.join(val_dir, "audio", filename)
                librosa.output.write_wav(output_path, audio, sr)
                val_samples.append(sample)
        
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    
    # Save metadata
    train_metadata = {"samples": train_samples}
    val_metadata = {"samples": val_samples}
    
    with open(os.path.join(train_dir, "metadata.json"), "w") as f:
        json.dump(train_metadata, f, indent=2)
    
    with open(os.path.join(val_dir, "metadata.json"), "w") as f:
        json.dump(val_metadata, f, indent=2)
    
    print(f"Prepared {len(train_samples)} training samples and {len(val_samples)} validation samples")


class VocalGenerationConfig:
    
    """Usage Examples
    `Training the Model
    bash
    python vocal_generator.py train \
        --data_dir data/vocals \
        --batch_size 16 \
        --learning_rate 1e-4 \
        --epochs 100 \
        --checkpoint_dir checkpoints/vocal_generator
    Generating Vocals
    bash
    python vocal_generator.py generate \
        --lyrics "I'm walking down the street on a sunny day" \
        --prompt "A cheerful pop vocal with female voice" \
        --gender 2 \
        --tempo 120 \
        --pitch 0 \
        --duration 10 \
        --checkpoint checkpoints/vocal_generator/checkpoint_epoch_100.pt \
        --output_file output/vocals/cheerful_pop.wav
    Interactive Mode
    bash
    python vocal_generator.py interactive \
        --checkpoint checkpoints/vocal_generator/checkpoint_epoch_100.pt`"""
    
    def __init__(self):
        # Model architecture
        self.model_name = "facebook/musicgen-melody"  # Base model for audio features
        self.lyric_encoder_model = "bert-base-uncased"  # For lyrics encoding
        self.hidden_size = 1024
        self.n_layers = 12
        self.n_heads = 16
        self.dropout = 0.1
        self.max_seq_len = 1000
        
        # Audio parameters
        self.sample_rate = 24000
        self.n_fft = 1024
        self.hop_length = 256
        self.n_mels = 80
        self.f_min = 0
        self.f_max = 8000
        
        # Training parameters
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.max_epochs = 100
        self.warmup_steps = 1000
        self.gradient_accumulation_steps = 1
        self.fp16 = True
        
        # Generation parameters
        self.max_duration = 30  # in seconds
        self.temperature = 1.0
        self.top_k = 50
        self.top_p = 0.95
        
        # Paths
        self.checkpoint_dir = "checkpoints/vocal_generator"
        self.data_dir = "data/vocals"
        self.output_dir = "output/vocals"
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

class LyricEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.lyric_encoder_model)
        self.encoder = AutoModel.from_pretrained(config.lyric_encoder_model)
        self.projection = nn.Linear(self.encoder.config.hidden_size, config.hidden_size)
        
    def forward(self, lyrics):
        # Tokenize lyrics
        inputs = self.tokenizer(lyrics, padding=True, truncation=True, 
                               max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        
        # Get embeddings
        outputs = self.encoder(**inputs)
        # Use sequence output (last hidden states)
        embeddings = outputs.last_hidden_state
        
        # Project to model dimension
        projected = self.projection(embeddings)
        
        return projected, inputs.attention_mask

class VocalStyleEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config.model_name)
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, config.hidden_size)
        
    def forward(self, audio_samples):
        if audio_samples is None:
            # Return zero embeddings if no audio sample provided
            batch_size = 1  # Default batch size when no samples
            return torch.zeros(batch_size, 1, self.config.hidden_size).to(self.encoder.device)
        
        # Process audio samples
        features = self.feature_extractor(audio_samples, 
                                         sampling_rate=self.config.sample_rate,
                                         return_tensors="pt")
        features = {k: v.to(self.encoder.device) for k, v in features.items()}
        
        # Get embeddings
        outputs = self.encoder(**features)
        # Use pooled output
        embeddings = outputs.pooler_output.unsqueeze(1)  # Add sequence dimension
        
        # Project to model dimension
        projected = self.projection(embeddings)
        
        return projected

class PromptEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.lyric_encoder_model)
        self.encoder = AutoModel.from_pretrained(config.lyric_encoder_model)
        self.projection = nn.Linear(self.encoder.config.hidden_size, config.hidden_size)
        
    def forward(self, prompts):
        # Tokenize prompts
        inputs = self.tokenizer(prompts, padding=True, truncation=True, 
                               max_length=256, return_tensors="pt")
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        
        # Get embeddings
        outputs = self.encoder(**inputs)
        # Use pooled output
        embeddings = outputs.pooler_output.unsqueeze(1)  # Add sequence dimension
        
        # Project to model dimension
        projected = self.projection(embeddings)
        
        return projected

class MusicAttributesEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings for categorical attributes
        self.gender_embedding = nn.Embedding(3, 64)  # male, female, neutral
        self.accent_embedding = nn.Embedding(10, 64)  # 10 different accents
        self.style_embedding = nn.Embedding(20, 128)  # 20 different styles
        
        # Projections for numerical attributes
        self.tempo_projection = nn.Linear(1, 64)
        self.pitch_projection = nn.Linear(1, 64)
        self.duration_projection = nn.Linear(1, 64)
        
        # Final projection
        self.projection = nn.Linear(64*4 + 128, config.hidden_size)
        
    def forward(self, attributes):
        # Extract attributes
        gender = attributes.get('gender', torch.zeros(1, dtype=torch.long).to(self.gender_embedding.weight.device))
        accent = attributes.get('accent', torch.zeros(1, dtype=torch.long).to(self.accent_embedding.weight.device))
        style = attributes.get('style', torch.zeros(1, dtype=torch.long).to(self.style_embedding.weight.device))
        tempo = attributes.get('tempo', torch.zeros(1, 1).to(self.tempo_projection.weight.device))
        pitch = attributes.get('pitch', torch.zeros(1, 1).to(self.pitch_projection.weight.device))
        duration = attributes.get('duration', torch.zeros(1, 1).to(self.duration_projection.weight.device))
        
        # Get embeddings
        gender_emb = self.gender_embedding(gender)
        accent_emb = self.accent_embedding(accent)
        style_emb = self.style_embedding(style)
        
        # Project numerical attributes
        tempo_emb = self.tempo_projection(tempo)
        pitch_emb = self.pitch_projection(pitch)
        duration_emb = self.duration_projection(duration)
        
        # Concatenate all embeddings
        combined = torch.cat([
            gender_emb, accent_emb, style_emb, 
            tempo_emb, pitch_emb, duration_emb
        ], dim=1)
        
        # Final projection
        projected = self.projection(combined).unsqueeze(1)  # Add sequence dimension
        
        return projected

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layer for audio tokens
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
        
        # Transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, config.n_layers)
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, tokens, memory, memory_mask=None):
        batch_size, seq_len = tokens.shape
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeddings = self.token_embedding(tokens)
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        
        # Create causal mask for autoregressive generation
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(tokens.device)
        
        # Pass through transformer
        outputs = self.transformer(
            tgt=embeddings, 
            memory=memory,
            tgt_mask=causal_mask,
            memory_mask=memory_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(outputs)
        
        return logits
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class VocalGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Add vocab size to config
        config.vocab_size = 8192  # Typical codebook size for audio tokens
        
        # Encoders
        self.lyric_encoder = LyricEncoder(config)
        self.style_encoder = VocalStyleEncoder(config)
        self.prompt_encoder = PromptEncoder(config)
        self.attributes_encoder = MusicAttributesEncoder(config)
        
        # Decoder
        self.decoder = TransformerDecoder(config)
        
        # Audio codec for tokenization and detokenization
        self.audio_processor = AutoProcessor.from_pretrained(config.model_name)
        
    def forward(self, lyrics, prompt, audio_sample=None, attributes=None, target_tokens=None):
        # Encode lyrics
        lyric_embeddings, lyric_mask = self.lyric_encoder(lyrics)
        
        # Encode prompt
        prompt_embeddings = self.prompt_encoder(prompt)
        
        # Encode audio sample if provided
        style_embeddings = self.style_encoder(audio_sample)
        
        # Encode attributes if provided
        if attributes is None:
            attributes = {}
        attribute_embeddings = self.attributes_encoder(attributes)
        
        # Combine all embeddings as memory for decoder
        memory = torch.cat([
            prompt_embeddings,
            style_embeddings,
            attribute_embeddings,
            lyric_embeddings
        ], dim=1)
        
        # Create memory mask
        memory_mask = torch.ones(memory.shape[0], memory.shape[1], device=memory.device)
        # Set mask for lyrics based on lyric_mask
        memory_mask[:, -lyric_embeddings.shape[1]:] = lyric_mask
        
        # If target tokens are provided (training), compute loss
        if target_tokens is not None:
            # Shift target tokens for teacher forcing (input = target shifted right)
            input_tokens = torch.cat([
                torch.zeros(target_tokens.shape[0], 1, dtype=torch.long, device=target_tokens.device),
                target_tokens[:, :-1]
            ], dim=1)
            
            # Forward pass through decoder
            logits = self.decoder(input_tokens, memory, memory_mask)
            
            # Compute loss
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), target_tokens.reshape(-1))
            
            return loss, logits
        
        # If no target tokens (inference), generate autoregressively
        else:
            generated_tokens = self.generate(memory, memory_mask)
            return generated_tokens
    
    def generate(self, memory, memory_mask=None, max_length=1000, temperature=1.0, top_k=50, top_p=0.95):
        batch_size = memory.shape[0]
        
        # Start with a single zero token
        tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=memory.device)
        
        # Generate tokens autoregressively
        for i in range(max_length - 1):
            # Forward pass through decoder
            logits = self.decoder(tokens, memory, memory_mask)
            
            # Get next token logits (last position)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Concatenate with previous tokens
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Check if all sequences have reached the end token (0 for simplicity)
            if (next_token == 0).all():
                break
        
        return tokens
    
    def tokenize_audio(self, audio_waveform, sample_rate=None):
        """Convert audio waveform to tokens using the audio codec"""
        if sample_rate is not None and sample_rate != self.config.sample_rate:
            # Resample audio if needed
            audio_waveform = librosa.resample(
                audio_waveform, 
                orig_sr=sample_rate, 
                target_sr=self.config.sample_rate
            )
        
        # Process audio with codec
        inputs = self.audio_processor(
            audio_waveform, 
            sampling_rate=self.config.sample_rate,
            return_tensors="pt"
        )
        
        # Extract tokens (implementation depends on specific codec)
        # This is a placeholder - actual implementation will depend on the codec used
        tokens = inputs["input_values"]  # Simplified
        
        return tokens
    
    def detokenize_audio(self, tokens):
        """Convert tokens back to audio waveform using the audio codec"""
        # This is a placeholder - actual implementation will depend on the codec used
        # In practice, you would use the codec's decoder to convert tokens to waveform
        waveform = tokens  # Simplified
        
        return waveform

class VocalDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split
        self.data_dir = Path(config.data_dir) / split
        
        # Load metadata
        with open(self.data_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)
        
        # Filter out invalid samples
        self.samples = [s for s in self.metadata["samples"] if self._is_valid_sample(s)]
        print(f"Loaded {len(self.samples)} {split} samples")
    
    def _is_valid_sample(self, sample):
        # Check if all required files exist
        audio_path = self.data_dir / "audio" / sample["audio_file"]
        if not audio_path.exists():
            return False
        
        # Check if duration is within limits
        if sample.get("duration", 0) > self.config.max_duration:
            return False
        
        return True
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio
        audio_path = self.data_dir / "audio" / sample["audio_file"]
        audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
        
        # Get lyrics
        lyrics = sample.get("lyrics", "")
        
        # Get prompt
        prompt = sample.get("prompt", "")
        
        # Get attributes
        attributes = {
            "gender": torch.tensor([sample.get("gender", 0)], dtype=torch.long),
            "accent": torch.tensor([sample.get("accent", 0)], dtype=torch.long),
            "style": torch.tensor([sample.get("style", 0)], dtype=torch.long),
            "tempo": torch.tensor([[sample.get("tempo", 120) / 200.0]]),  # Normalize tempo
            "pitch": torch.tensor([[sample.get("pitch", 0) / 12.0]]),  # Normalize pitch
            "duration": torch.tensor([[min(sample.get("duration", 30), self.config.max_duration) / self.config.max_duration]])
        }
        
        # Get reference audio if available
        ref_audio = None
        if "reference_file" in sample and sample["reference_file"]:
            ref_path = self.data_dir / "references" / sample["reference_file"]
            if ref_path.exists():
                ref_audio, ref_sr = librosa.load(ref_path, sr=self.config.sample_rate)
        
        # Tokenize audio (placeholder - actual implementation depends on codec)
        # In practice, you might precompute and store tokens instead of computing on-the-fly
        tokens = torch.randint(0, self.config.vocab_size, (min(int(len(audio) / self.config.hop_length), 1000),))
        
        return {
            "audio": audio,
            "tokens": tokens,
            "lyrics": lyrics,
            "prompt": prompt,
            "attributes": attributes,
            "reference_audio": ref_audio
        }
    
    def collate_fn(self, batch):
        # Collate function for DataLoader
        max_token_len = max(len(item["tokens"]) for item in batch)
        
        # Pad tokens
        padded_tokens = torch.zeros(len(batch), max_token_len, dtype=torch.long)
        for i, item in enumerate(batch):
            padded_tokens[i, :len(item["tokens"])] = item["tokens"]
        
        # Collect other items
        lyrics = [item["lyrics"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        
        # Collect attributes
        attributes = {
            key: torch.cat([item["attributes"][key] for item in batch], dim=0)
            for key in batch[0]["attributes"]
        }
        
        # Collect reference audio (if available)
        reference_audio = [item["reference_audio"] for item in batch]
        if all(audio is None for audio in reference_audio):
            reference_audio = None
        
        return {
            "tokens": padded_tokens,
            "lyrics": lyrics,
            "prompt": prompts,
            "attributes": attributes,
            "reference_audio": reference_audio
        }

def train(config, model, train_dataset, val_dataset=None, resume_from=None):
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=4
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
            num_workers=4
        )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs * len(train_loader),
        eta_min=config.learning_rate / 10
    )
    
    # Move model to device
    model = model.to(config.device)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=config.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(start_epoch, config.max_epochs):
        model.train()
        train_loss = 0
        
        # Progress bar
        progress = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.max_epochs}")
        
        for batch in progress:
            # Move batch to device
            tokens = batch["tokens"].to(config.device)
            
            # Forward pass
            loss, _ = model(
                lyrics=batch["lyrics"],
                prompt=batch["prompt"],
                audio_sample=batch["reference_audio"],
                attributes=batch["attributes"],
                target_tokens=tokens
            )
            
            # Backward pass
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            # Update weights if gradient accumulation steps reached
            if (progress.n + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update progress bar
            train_loss += loss.item() * config.gradient_accumulation_steps
            progress.set_postfix({"loss": train_loss / (progress.n + 1)})
        
        # Validation
        if val_dataset:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm.tqdm(val_loader, desc="Validation"):
                    # Move batch to device
                    tokens = batch["tokens"].to(config.device)
                    
                    # Forward pass
                    loss, _ = model(
                        lyrics=batch["lyrics"],
                        prompt=batch["prompt"],
                        audio_sample=batch["reference_audio"],
                        attributes=batch["attributes"],
                        target_tokens=tokens
                    )
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            print(f"Validation loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config.__dict__
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

def generate_vocals(
    model, 
    lyrics, 
    prompt, 
    reference_audio=None, 
    gender=0,  # 0=neutral, 1=male, 2=female
    accent=0,
    style=0,
    tempo=120,
    pitch=0,
    duration=30,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    output_file=None
):
    """Generate vocals using the trained model"""
    # Prepare attributes
    attributes = {
        "gender": torch.tensor([gender], dtype=torch.long).to(model.device),
        "accent": torch.tensor([accent], dtype=torch.long).to(model.device),
        "style": torch.tensor([style], dtype=torch.long).to(model.device),
        "tempo": torch.tensor([[tempo / 200.0]]).to(model.device),
        "pitch": torch.tensor([[pitch / 12.0]]).to(model.device),
        "duration": torch.tensor([[min(duration, model.config.max_duration) / model.config.max_duration]]).to(model.device)
    }
    
    # Process reference audio if provided
    if reference_audio is not None:
        if isinstance(reference_audio, str):
            # Load audio file
            audio, sr = librosa.load(reference_audio, sr=model.config.sample_rate)
            reference_audio = audio
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate tokens
    with torch.no_grad():
        # Encode inputs
        lyric_embeddings, lyric_mask = model.lyric_encoder([lyrics])
        prompt_embeddings = model.prompt_encoder([prompt])
        style_embeddings = model.style_encoder(reference_audio)
        attribute_embeddings = model.attributes_encoder(attributes)
        
        # Combine embeddings
        memory = torch.cat([
            prompt_embeddings,
            style_embeddings,
            attribute_embeddings,
            lyric_embeddings
        ], dim=1)
        
        # Create memory mask
        memory_mask = torch.ones(memory.shape[0], memory.shape[1], device=memory.device)
        memory_mask[:, -lyric_embeddings.shape[1]:] = lyric_mask
        
        # Generate tokens
        tokens = model.generate(
            memory, 
            memory_mask, 
            max_length=int(duration * model.config.sample_rate / model.config.hop_length),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    # Convert tokens to audio
    audio = model.detokenize_audio(tokens[0])
    
    # Save audio if output file specified
    if output_file:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save audio
        librosa.output.write_wav(output_file, audio, model.config.sample_rate)
        print(f"Saved generated vocals to {output_file}")
    
    return audio, tokens[0]

def main():
    parser = argparse.ArgumentParser(description="Vocal Generation System")
    subparsers = parser.add_subparsers(dest="mode", help="Mode")
    
    # Training parser
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data_dir", type=str, default="data/vocals", help="Data directory")
    train_parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/vocal_generator", help="Checkpoint directory")
    train_parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    
    # Generation parser
    gen_parser = subparsers.add_parser("generate", help="Generate vocals")
    gen_parser.add_argument("--lyrics", type=str, required=True, help="Lyrics for the vocals")
    gen_parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing the desired vocals")
    gen_parser.add_argument("--reference_audio", type=str, default=None, help="Reference audio file for style")
    gen_parser.add_argument("--gender", type=int, default=0, help="Gender (0=neutral, 1=male, 2=female)")
    gen_parser.add_argument("--accent", type=int, default=0, help="Accent (0-9)")
    gen_parser.add_argument("--style", type=int, default=0, help="Style (0-19)")
    gen_parser.add_argument("--tempo", type=float, default=120, help="Tempo in BPM")
    gen_parser.add_argument("--pitch", type=int, default=0, help="Pitch adjustment in semitones")
    gen_parser.add_argument("--duration", type=float, default=30, help="Duration in seconds")
    gen_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    gen_parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    gen_parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    gen_parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    gen_parser.add_argument("--output_file", type=str, default="output/vocals/generated.wav", help="Output file")
    
    # Interactive parser
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    interactive_parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    
    args = parser.parse_args()
    
    # Create config
    config = VocalGenerationConfig()
    
    if args.mode == "train":
        # Update config with command line arguments
        config.data_dir = args.data_dir
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate
        config.max_epochs = args.epochs
        config.checkpoint_dir = args.checkpoint_dir
        
        # Create datasets
        train_dataset = VocalDataset(config, split="train")
        val_dataset = VocalDataset(config, split="val")
        
        # Create model
        model = VocalGenerator(config)
        
        # Train model
        train(config, model, train_dataset, val_dataset, resume_from=args.resume_from)
    
    elif args.mode == "generate":
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        
        # Update config from checkpoint
        for key, value in checkpoint["config"].items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Create model
        model = VocalGenerator(config)
        model.load_state_dict(checkpoint["model"])
        model = model.to(config.device)
        
        # Generate vocals
        generate_vocals(
            model=model,
            lyrics=args.lyrics,
            prompt=args.prompt,
            reference_audio=args.reference_audio,
            gender=args.gender,
            accent=args.accent,
            style=args.style,
            tempo=args.tempo,
            pitch=args.pitch,
            duration=args.duration,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            output_file=args.output_file
        )
    
    elif args.mode == "interactive":
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        
        # Update config from checkpoint
        for key, value in checkpoint["config"].items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Create model
        model = VocalGenerator(config)
        model.load_state_dict(checkpoint["model"])
        model = model.to(config.device)
        
        # Interactive mode
        interactive_generation(model)

def interactive_generation(model):
    """Interactive mode for vocal generation"""
    print("=== Vocal Generation Interactive Mode ===")
    print("Enter 'q' to quit")
    
    while True:
        # Get lyrics
        lyrics = input("\nEnter lyrics: ")
        if lyrics.lower() == 'q':
            break
        
        # Get prompt
        prompt = input("Enter prompt: ")
        if prompt.lower() == 'q':
            break
        
        # Get reference audio
        reference_audio = input("Enter reference audio file (optional): ")
        if reference_audio.lower() == 'q':
            break
        if not reference_audio:
            reference_audio = None
        
        # Get gender
        gender_str = input("Enter gender (neutral/male/female, default=neutral): ")
        if gender_str.lower() == 'q':
            break
        gender = 0  # neutral
        if gender_str.lower() == "male":
            gender = 1
        elif gender_str.lower() == "female":
            gender = 2
        
        # Get tempo
        tempo_str = input("Enter tempo in BPM (default=120): ")
        if tempo_str.lower() == 'q':
            break
        tempo = 120
        if tempo_str:
            try:
                tempo = float(tempo_str)
            except ValueError:
                print("Invalid tempo, using default")
        
        # Get pitch
        pitch_str = input("Enter pitch adjustment in semitones (default=0): ")
        if pitch_str.lower() == 'q':
            break
        pitch = 0
        if pitch_str:
            try:
                pitch = int(pitch_str)
            except ValueError:
                print("Invalid pitch, using default")
        
        # Get duration
        duration_str = input("Enter duration in seconds (default=30): ")
        if duration_str.lower() == 'q':
            break
        duration = 30
        if duration_str:
            try:
                duration = float(duration_str)
            except ValueError:
                print("Invalid duration, using default")
        
        # Get output file
        output_file = input("Enter output file (default=output/vocals/generated.wav): ")
        if output_file.lower() == 'q':
            break
        if not output_file:
            output_file = "output/vocals/generated.wav"
        
        # Generate vocals
        print("\nGenerating vocals...")
        generate_vocals(
            model=model,
            lyrics=lyrics,
            prompt=prompt,
            reference_audio=reference_audio,
            gender=gender,
            tempo=tempo,
            pitch=pitch,
            duration=duration,
            output_file=output_file
        )
        print("Done!")

if __name__ == "__main__":
    main()
