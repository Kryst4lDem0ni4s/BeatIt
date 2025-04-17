import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pickle
import random
import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import librosa

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('cmudict')

def prepare_dataset(
    lyrics_dir,
    audio_dir=None,
    output_dir="data/lyrics", 
    split_ratio=0.9
):
    """Prepare dataset from raw lyrics and audio files"""
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    for directory in [
        train_dir, val_dir,
        os.path.join(train_dir, "lyrics"),
        os.path.join(train_dir, "audio"),
        os.path.join(val_dir, "lyrics"),
        os.path.join(val_dir, "audio")
    ]:
        os.makedirs(directory, exist_ok=True)
    
    # Find all lyrics files
    lyrics_files = []
    for root, _, files in os.walk(lyrics_dir):
        for file in files:
            if file.endswith((".txt", ".lrc")):
                lyrics_files.append(os.path.join(root, file))
    
    print(f"Found {len(lyrics_files)} lyrics files")
    
    # Process each file
    train_samples = []
    val_samples = []
    
    for i, lyrics_path in enumerate(tqdm.tqdm(lyrics_files)):
        try:
            # Load lyrics
            with open(lyrics_path, "r", encoding="utf-8") as f:
                lyrics = f.read().strip()
            
            # Skip if lyrics are too short
            if len(lyrics.split()) < 10:
                continue
            
            # Create sample metadata
            filename = f"{i:06d}.txt"
            sample = {
                "lyrics_file": filename,
                "original_path": lyrics_path
            }
            
            # Look for matching audio file if audio_dir is provided
            if audio_dir:
                lyrics_basename = os.path.splitext(os.path.basename(lyrics_path))[0]
                potential_audio_files = [
                    os.path.join(audio_dir, f"{lyrics_basename}.mp3"),
                    os.path.join(audio_dir, f"{lyrics_basename}.wav"),
                    os.path.join(audio_dir, f"{lyrics_basename}.ogg")
                ]
                
                for audio_path in potential_audio_files:
                    if os.path.exists(audio_path):
                        audio_filename = f"{i:06d}{os.path.splitext(audio_path)[1]}"
                        sample["audio_file"] = audio_filename
                        
                        # Copy audio file
                        if random.random() < split_ratio:
                            output_audio_path = os.path.join(train_dir, "audio", audio_filename)
                        else:
                            output_audio_path = os.path.join(val_dir, "audio", audio_filename)
                        
                        import shutil
                        shutil.copy2(audio_path, output_audio_path)
                        break
            
            # Extract potential style/themes from lyrics content
            # This is a simplified approach - in practice, you'd use a more sophisticated method
            common_themes = ["love", "heartbreak", "party", "life", "dreams", "hope", "sadness"]
            sample_themes = []
            for theme in common_themes:
                if theme in lyrics.lower():
                    sample_themes.append(theme)
            
            if sample_themes:
                sample["style_themes"] = sample_themes
            
            # Decide if sample goes to train or validation set
            if random.random() < split_ratio:
                # Save lyrics file to train directory
                output_path = os.path.join(train_dir, "lyrics", filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(lyrics)
                train_samples.append(sample)
            else:
                # Save lyrics file to validation directory
                output_path = os.path.join(val_dir, "lyrics", filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(lyrics)
                val_samples.append(sample)
        
        except Exception as e:
            print(f"Error processing {lyrics_path}: {e}")
    
    # Save metadata
    train_metadata = {"samples": train_samples}
    val_metadata = {"samples": val_samples}
    
    with open(os.path.join(train_dir, "metadata.json"), "w") as f:
        json.dump(train_metadata, f, indent=2)
    
    with open(os.path.join(val_dir, "metadata.json"), "w") as f:
        json.dump(val_metadata, f, indent=2)
    
    print(f"Prepared {len(train_samples)} training samples and {len(val_samples)} validation samples")


class LyricsGenerationConfig:
    """Usage Examples
    Training the Model
    bash
    python lyrics_generator.py train \
        --data_dir data/lyrics \
        --batch_size 8 \
        --learning_rate 5e-5 \
        --epochs 10 \
        --checkpoint_dir checkpoints/lyrics_generator
    Generating Lyrics
    bash
    python lyrics_generator.py generate \
        --prompt "Write a melancholic song about lost love" \
        --audio_path samples/reference.mp3 \
        --tempo 90 \
        --style_themes "sad, emotional, ballad" \
        --checkpoint checkpoints/lyrics_generator/checkpoint_epoch_10.pt \
        --output_file output/lyrics/lost_love.txt
    Interactive Mode
    bash
    python lyrics_generator.py interactive \
    --checkpoint checkpoints/lyrics_generator/checkpoint_epoch_10.pt"""
    
    
    def __init__(self):
        # Model architecture
        self.model_name = "gpt2-medium"  # Base model for lyrics generation
        self.style_encoder_model = "sentence-transformers/all-mpnet-base-v2"  # For style encoding
        self.hidden_size = 1024
        self.max_seq_len = 512
        
        # Audio parameters
        self.sample_rate = 22050
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        
        # Training parameters
        self.batch_size = 8
        self.learning_rate = 5e-5
        self.weight_decay = 0.01
        self.max_epochs = 10
        self.warmup_steps = 500
        self.gradient_accumulation_steps = 4
        
        # Generation parameters
        self.max_length = 256
        self.temperature = 0.9
        self.top_k = 50
        self.top_p = 0.95
        self.repetition_penalty = 1.2
        
        # Paths
        self.checkpoint_dir = "checkpoints/lyrics_generator"
        self.data_dir = "data/lyrics"
        self.output_dir = "output/lyrics"
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

class AudioEncoder(nn.Module):
    """Encodes audio samples into style embeddings"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # CNN layers for audio processing
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the size after convolutions and pooling
        self.fc_input_size = self._calculate_conv_output_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, config.hidden_size)
        self.dropout = nn.Dropout(0.3)
        
    def _calculate_conv_output_size(self):
        # Calculate the output size of the convolutional layers
        # This is a placeholder - actual calculation depends on your input size
        return 128 * 16 * 16  # Example value
    
    def forward(self, audio_features):
        # Input shape: [batch_size, 1, n_mels, time]
        x = F.relu(self.conv1(audio_features))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class StyleEncoder(nn.Module):
    """Encodes style/theme text descriptions into embeddings"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.style_encoder_model)
        self.encoder = AutoModel.from_pretrained(config.style_encoder_model)
        
    def forward(self, text_list):
        # Tokenize the input texts
        inputs = self.tokenizer(text_list, padding=True, truncation=True, 
                               max_length=128, return_tensors="pt")
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Use mean pooling to get a single vector per text
            attention_mask = inputs['attention_mask']
            embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
            
        return embeddings
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        # Mean pooling - take mean of all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class TempoEncoder(nn.Module):
    """Encodes tempo information into embeddings"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tempo_embedding = nn.Embedding(300, 64)  # Embedding for tempos from 30-300 BPM
        self.projection = nn.Linear(64, config.hidden_size)
        
    def forward(self, tempo):
        # Clamp tempo to valid range and convert to long tensor
        tempo_clamped = torch.clamp(tempo, min=30, max=300).long() - 30  # Offset by 30 to start from 0
        
        # Get embeddings
        tempo_emb = self.tempo_embedding(tempo_clamped)
        projected = self.projection(tempo_emb)
        
        return projected

class SyllableAnalyzer:
    """Analyzes syllable patterns in text and music"""
    def __init__(self):
        self.cmu_dict = cmudict.dict()
        
    def count_syllables(self, word):
        """Count syllables in a word using CMU dictionary"""
        word = word.lower()
        if word in self.cmu_dict:
            return max([len([y for y in x if y[-1].isdigit()]) for x in self.cmu_dict[word]])
        else:
            # Fallback method for words not in the dictionary
            return self._count_syllables_fallback(word)
    
    def _count_syllables_fallback(self, word):
        """Fallback syllable counter for words not in CMU dict"""
        vowels = "aeiouy"
        word = word.lower()
        count = 0
        if word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count = 1
        return count
    
    def analyze_text_syllables(self, text):
        """Analyze syllable pattern in a text"""
        words = word_tokenize(text)
        syllable_counts = [self.count_syllables(word) for word in words if word.isalpha()]
        return syllable_counts
    
    def extract_syllable_template(self, audio, sr, hop_length):
        """Extract syllable template from audio using onset detection"""
        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
        
        # Pick peaks (onsets)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        
        # Convert frame indices to time
        onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)
        
        # Estimate syllable count based on onset density
        # This is a simplified approach - real implementation would be more sophisticated
        if len(onset_times) > 0:
            # Calculate time differences between consecutive onsets
            time_diffs = np.diff(onset_times, prepend=0)
            
            # Filter out very short differences (less than 0.1 seconds)
            valid_diffs = time_diffs[time_diffs > 0.1]
            
            # Estimate syllable count based on valid onsets
            syllable_count = len(valid_diffs) + 1
        else:
            syllable_count = 1
            
        return syllable_count

class LyricsGenerator(nn.Module):
    """Main lyrics generation model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load base language model
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(config.model_name)
        
        # Add special tokens if needed
        special_tokens = {
            "pad_token": "<|pad|>",
            "bos_token": "<|startoftext|>",
            "eos_token": "<|endoftext|>",
            "additional_special_tokens": ["<|verse|>", "<|chorus|>", "<|bridge|>"]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Encoders for different input types
        self.audio_encoder = AudioEncoder(config)
        self.style_encoder = StyleEncoder(config)
        self.tempo_encoder = TempoEncoder(config)
        
        # Syllable analyzer
        self.syllable_analyzer = SyllableAnalyzer()
        
        # Projection layer to combine all conditioning
        self.conditioning_projection = nn.Linear(config.hidden_size * 3, config.hidden_size)
        
    def encode_audio(self, audio_path):
        """Extract features from audio file and encode them"""
        if audio_path is None:
            # Return zero tensor if no audio provided
            return torch.zeros(1, self.config.hidden_size).to(self.model.device)
        
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_fft=self.config.n_fft, 
            hop_length=self.config.hop_length, 
            n_mels=self.config.n_mels
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        # Normalize
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-9)
        
        # Add batch and channel dimensions
        log_mel_spec = torch.tensor(log_mel_spec).unsqueeze(0).unsqueeze(0).to(self.model.device)
        
        # Encode
        audio_embedding = self.audio_encoder(log_mel_spec)
        
        return audio_embedding
    
    def prepare_conditioning(self, prompt, audio_path=None, sample_lyrics=None, tempo=120, style_themes=None):
        """Prepare conditioning inputs for the model"""
        # Encode audio if provided
        if audio_path:
            audio_embedding = self.encode_audio(audio_path)
        else:
            audio_embedding = torch.zeros(1, self.config.hidden_size).to(self.model.device)
        
        # Encode style/themes if provided
        if style_themes:
            if isinstance(style_themes, str):
                style_themes = [style_themes]
            style_embedding = self.style_encoder(style_themes)
        else:
            style_embedding = torch.zeros(1, self.config.hidden_size).to(self.model.device)
        
        # Encode tempo
        tempo_tensor = torch.tensor([tempo]).to(self.model.device)
        tempo_embedding = self.tempo_encoder(tempo_tensor)
        
        # Combine all conditioning
        combined_embedding = torch.cat([
            audio_embedding, 
            style_embedding, 
            tempo_embedding
        ], dim=1)
        
        # Project to model dimension
        conditioning = self.conditioning_projection(combined_embedding)
        
        # Prepare input text
        if sample_lyrics:
            input_text = f"{self.tokenizer.bos_token} {prompt} {sample_lyrics} {self.tokenizer.eos_token}"
        else:
            input_text = f"{self.tokenizer.bos_token} {prompt} {self.tokenizer.eos_token}"
        
        return conditioning, input_text
    
    def forward(self, input_ids, attention_mask, conditioning=None):
        """Forward pass through the model"""
        # Get base model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # If conditioning is provided, apply it
        if conditioning is not None:
            # Add conditioning to the hidden states
            hidden_states = outputs.hidden_states[-1]
            conditioned_hidden_states = hidden_states + conditioning.unsqueeze(1)
            
            # Compute new logits
            logits = self.model.lm_head(conditioned_hidden_states)
        else:
            logits = outputs.logits
        
        return logits
    
    def generate_lyrics(self, prompt, audio_path=None, sample_lyrics=None, tempo=120, style_themes=None, 
                       max_length=None, temperature=None, top_k=None, top_p=None, repetition_penalty=None):
        """Generate lyrics based on inputs"""
        # Set default generation parameters if not provided
        max_length = max_length or self.config.max_length
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k
        top_p = top_p or self.config.top_p
        repetition_penalty = repetition_penalty or self.config.repetition_penalty
        
        # Prepare conditioning
        conditioning, input_text = self.prepare_conditioning(
            prompt, audio_path, sample_lyrics, tempo, style_themes
        )
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        
        # Set the model to evaluation mode
        self.model.eval()
        
        # Generate lyrics
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Remove the prompt from the generated text
        if prompt in generated_text:
            generated_text = generated_text.split(prompt, 1)[1].strip()
        
        # Remove sample lyrics from the generated text if they were provided
        if sample_lyrics and sample_lyrics in generated_text:
            generated_text = generated_text.split(sample_lyrics, 1)[1].strip()
        
        return generated_text

class LyricsDataset(Dataset):
    """Dataset for training the lyrics generator"""
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split
        self.data_dir = os.path.join(config.data_dir, split)
        
        # Load metadata
        with open(os.path.join(self.data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        
        # Filter out invalid samples
        self.samples = [s for s in self.metadata["samples"] if self._is_valid_sample(s)]
        print(f"Loaded {len(self.samples)} {split} samples")
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
        
        # Add special tokens if needed
        special_tokens = {
            "pad_token": "<|pad|>",
            "bos_token": "<|startoftext|>",
            "eos_token": "<|endoftext|>",
            "additional_special_tokens": ["<|verse|>", "<|chorus|>", "<|bridge|>"]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Initialize syllable analyzer
        self.syllable_analyzer = SyllableAnalyzer()
    
    def _is_valid_sample(self, sample):
        """Check if sample is valid"""
        # Check if lyrics file exists
        lyrics_path = os.path.join(self.data_dir, "lyrics", sample["lyrics_file"])
        if not os.path.exists(lyrics_path):
            return False
        
        # Check if audio file exists if specified
        if "audio_file" in sample:
            audio_path = os.path.join(self.data_dir, "audio", sample["audio_file"])
            if not os.path.exists(audio_path):
                return False
        
        return True
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load lyrics
        lyrics_path = os.path.join(self.data_dir, "lyrics", sample["lyrics_file"])
        with open(lyrics_path, "r", encoding="utf-8") as f:
            lyrics = f.read().strip()
        
        # Get prompt
        prompt = sample.get("prompt", "")
        
        # Get style/themes
        style_themes = sample.get("style_themes", [])
        
        # Get tempo
        tempo = sample.get("tempo", 120)
        
        # Load audio if available
        audio_path = None
        if "audio_file" in sample:
            audio_path = os.path.join(self.data_dir, "audio", sample["audio_file"])
        
        # Prepare input text
        input_text = f"{self.tokenizer.bos_token} {prompt} {lyrics} {self.tokenizer.eos_token}"
        
        # Tokenize text
        tokenized = self.tokenizer(
            input_text,
            max_length=self.config.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Extract features
        input_ids = tokenized.input_ids[0]
        attention_mask = tokenized.attention_mask[0]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompt,
            "lyrics": lyrics,
            "style_themes": style_themes,
            "tempo": tempo,
            "audio_path": audio_path
        }
    
    def collate_fn(self, batch):
        """Collate function for DataLoader"""
        # Collect input_ids and attention_mask
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        
        # Collect other items
        prompts = [item["prompt"] for item in batch]
        lyrics = [item["lyrics"] for item in batch]
        style_themes = [item["style_themes"] for item in batch]
        tempos = torch.tensor([item["tempo"] for item in batch])
        audio_paths = [item["audio_path"] for item in batch]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompts": prompts,
            "lyrics": lyrics,
            "style_themes": style_themes,
            "tempos": tempos,
            "audio_paths": audio_paths
        }

def train(config, model, train_dataset, val_dataset=None, resume_from=None):
    """Train the lyrics generator model"""
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
    total_steps = len(train_loader) * config.max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
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
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Compute loss
            # Shift logits and labels for next token prediction
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_attention_mask = attention_mask[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Apply attention mask
            loss = loss.view(shift_labels.size())
            loss = loss * shift_attention_mask
            
            # Average loss
            loss = loss.sum() / shift_attention_mask.sum()
            
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
                    input_ids = batch["input_ids"].to(config.device)
                    attention_mask = batch["attention_mask"].to(config.device)
                    
                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    # Compute loss
                    # Shift logits and labels for next token prediction
                    shift_logits = outputs[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    shift_attention_mask = attention_mask[..., 1:].contiguous()
                    
                    # Flatten the tokens
                    loss_fct = nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    
                    # Apply attention mask
                    loss = loss.view(shift_labels.size())
                    loss = loss * shift_attention_mask
                    
                    # Average loss
                    loss = loss.sum() / shift_attention_mask.sum()
                    
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

def main():
    """Main function for training and generating lyrics"""
    parser = argparse.ArgumentParser(description="Lyrics Generation System")
    subparsers = parser.add_subparsers(dest="mode", help="Mode")
    
    # Training parser
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data_dir", type=str, default="data/lyrics", help="Data directory")
    train_parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/lyrics_generator", help="Checkpoint directory")
    train_parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    
    # Generation parser
    gen_parser = subparsers.add_parser("generate", help="Generate lyrics")
    gen_parser.add_argument("--prompt", type=str, required=True, help="Text prompt for lyrics generation")
    gen_parser.add_argument("--audio_path", type=str, default=None, help="Path to audio sample for style reference")
    gen_parser.add_argument("--sample_lyrics", type=str, default=None, help="Sample lyrics for reference")
    gen_parser.add_argument("--tempo", type=int, default=120, help="Tempo in BPM")
    gen_parser.add_argument("--style_themes", type=str, default=None, help="Style/themes for lyrics (comma-separated)")
    gen_parser.add_argument("--max_length", type=int, default=256, help="Maximum length of generated lyrics")
    gen_parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    gen_parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    gen_parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    gen_parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty")
    gen_parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    gen_parser.add_argument("--output_file", type=str, default=None, help="Output file (optional)")
    
    # Interactive parser
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    interactive_parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    
    args = parser.parse_args()
    
    # Create config
    config = LyricsGenerationConfig()
    
    if args.mode == "train":
        # Update config with command line arguments
        config.data_dir = args.data_dir
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate
        config.max_epochs = args.epochs
        config.checkpoint_dir = args.checkpoint_dir
        
        # Create datasets
        train_dataset = LyricsDataset(config, split="train")
        val_dataset = LyricsDataset(config, split="val")
        
        # Create model
        model = LyricsGenerator(config)
        
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
        model = LyricsGenerator(config)
        model.load_state_dict(checkpoint["model"])
        model = model.to(config.device)
        
        # Parse style themes
        style_themes = None
        if args.style_themes:
            style_themes = [theme.strip() for theme in args.style_themes.split(",")]
        
        # Generate lyrics
        generated_lyrics = model.generate_lyrics(
            prompt=args.prompt,
            audio_path=args.audio_path,
            sample_lyrics=args.sample_lyrics,
            tempo=args.tempo,
            style_themes=style_themes,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        
        # Print generated lyrics
        print("\nGenerated Lyrics:")
        print("-----------------")
        print(generated_lyrics)
        print("-----------------")
        
        # Save to file if specified
        if args.output_file:
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(generated_lyrics)
            print(f"Saved lyrics to {args.output_file}")
    
    elif args.mode == "interactive":
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        
        # Update config from checkpoint
        for key, value in checkpoint["config"].items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Create model
        model = LyricsGenerator(config)
        model.load_state_dict(checkpoint["model"])
        model = model.to(config.device)
        
        # Interactive mode
        interactive_generation(model)

def interactive_generation(model):
    """Interactive mode for lyrics generation"""
    print("=== Lyrics Generation Interactive Mode ===")
    print("Enter 'q' to quit")
    
    while True:
        # Get prompt
        prompt = input("\nEnter prompt: ")
        if prompt.lower() == 'q':
            break
        
        # Get sample lyrics
        sample_lyrics = input("Enter sample lyrics (optional): ")
        if sample_lyrics.lower() == 'q':
            break
        if not sample_lyrics:
            sample_lyrics = None
        
        # Get audio path
        audio_path = input("Enter audio file path (optional): ")
        if audio_path.lower() == 'q':
            break
        if not audio_path or not os.path.exists(audio_path):
            audio_path = None
        
        # Get tempo
        tempo_str = input("Enter tempo in BPM (default=120): ")
        if tempo_str.lower() == 'q':
            break
        tempo = 120
        if tempo_str:
            try:
                tempo = int(tempo_str)
            except ValueError:
                print("Invalid tempo, using default")
        
        # Get style themes
        style_themes_str = input("Enter style/themes (comma-separated, optional): ")
        if style_themes_str.lower() == 'q':
            break
        style_themes = None
        if style_themes_str:
            style_themes = [theme.strip() for theme in style_themes_str.split(",")]
        
        # Get generation parameters
        temperature_str = input("Enter temperature (default=0.9): ")
        if temperature_str.lower() == 'q':
            break
        temperature = 0.9
        if temperature_str:
            try:
                temperature = float(temperature_str)
            except ValueError:
                print("Invalid temperature, using default")
        
        # Generate lyrics
        print("\nGenerating lyrics...")
        generated_lyrics = model.generate_lyrics(
            prompt=prompt,
            audio_path=audio_path,
            sample_lyrics=sample_lyrics,
            tempo=tempo,
            style_themes=style_themes,
            temperature=temperature
        )
        
        # Print generated lyrics
        print("\nGenerated Lyrics:")
        print("-----------------")
        print(generated_lyrics)
        print("-----------------")
        
        # Ask if user wants to save
        save = input("\nSave lyrics to file? (y/n): ")
        if save.lower() == 'y':
            output_file = input("Enter output file path: ")
            if output_file:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(generated_lyrics)
                print(f"Saved lyrics to {output_file}")

if __name__ == "__main__":
    main()
