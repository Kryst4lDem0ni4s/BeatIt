import os
import argparse
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import json
import pickle
import tqdm
import random
import csv
from pathlib import Path
from transformers import AutoProcessor, MusicgenForConditionalGeneration, AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class MusicGenConfig:
    
    """python musicgen.py train_csv --csv_file training_data/udio_songs_20250418_132910.csv --data_dir training_data/musicgen --batch_size 4 --learning_rate 1e-5 --epochs 10 --checkpoint_dir checkpoints/musicgen"""
    def __init__(self):
        # Model architecture
        self.model_name = "facebook/musicgen-melody"  # Base model for music generation
        self.text_encoder_model = "sentence-transformers/all-mpnet-base-v2"  # For style encoding
        self.hidden_size = 1024
        self.max_seq_len = 1024
        
        # Audio parameters
        self.sample_rate = 32000
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        
        # Training parameters
        self.batch_size = 4
        self.learning_rate = 1e-5
        self.weight_decay = 0.01
        self.max_epochs = 10
        self.warmup_steps = 500
        self.gradient_accumulation_steps = 4
        self.fp16 = True
        
        # Generation parameters
        self.max_duration = 30  # in seconds
        self.conditioning_scale = 3.0
        self.guidance_scale = 3.0
        
        # Paths
        self.checkpoint_dir = "checkpoints/musicgen"
        self.data_dir = "data/music"
        self.output_dir = "output/music"
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class StyleEncoder(nn.Module):
    """Encodes style/theme text descriptions into embeddings"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_model)
        self.encoder = AutoModel.from_pretrained(config.text_encoder_model)
        
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


class MusicalAttributesEncoder(nn.Module):
    """Encodes musical attributes like tempo, pitch, etc."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings for categorical attributes
        self.instrument_embedding = nn.Embedding(100, 64)  # 100 different instruments
        self.style_embedding = nn.Embedding(50, 128)  # 50 different styles
        
        # Projections for numerical attributes
        self.tempo_projection = nn.Linear(1, 64)
        self.pitch_projection = nn.Linear(1, 64)
        self.duration_projection = nn.Linear(1, 64)
        
        # Final projection
        self.projection = nn.Linear(64*3 + 128 + 64, config.hidden_size)
        
    def forward(self, attributes):
        # Extract attributes
        instruments = attributes.get('instruments', torch.zeros(1, dtype=torch.long).to(self.instrument_embedding.weight.device))
        style = attributes.get('style', torch.zeros(1, dtype=torch.long).to(self.style_embedding.weight.device))
        tempo = attributes.get('tempo', torch.zeros(1, 1).to(self.tempo_projection.weight.device))
        pitch = attributes.get('pitch', torch.zeros(1, 1).to(self.pitch_projection.weight.device))
        duration = attributes.get('duration', torch.zeros(1, 1).to(self.duration_projection.weight.device))
        
        # Get embeddings
        instrument_emb = self.instrument_embedding(instruments)
        style_emb = self.style_embedding(style)
        
        # Project numerical attributes
        tempo_emb = self.tempo_projection(tempo)
        pitch_emb = self.pitch_projection(pitch)
        duration_emb = self.duration_projection(duration)
        
        # Combine embeddings
        # For simplicity, we're just summing the instrument embeddings
        if len(instrument_emb.shape) > 2:
            instrument_emb = instrument_emb.sum(dim=1)
        
        # Concatenate all embeddings
        combined = torch.cat([
            instrument_emb, style_emb, 
            tempo_emb, pitch_emb, duration_emb
        ], dim=1)
        
        # Final projection
        projected = self.projection(combined)
        
        return projected


class MusicDataset(Dataset):
    """Dataset for training the music generator"""
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
        
        # Initialize processor
        self.processor = AutoProcessor.from_pretrained(config.model_name)
    
    def _is_valid_sample(self, sample):
        """Check if sample is valid"""
        # Check if audio file exists
        audio_path = os.path.join(self.data_dir, "audio", sample["audio_file"])
        if not os.path.exists(audio_path):
            return False
        
        # Check if sample has a prompt
        if not sample.get("prompt"):
            return False
        
        return True
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio
        audio_path = os.path.join(self.data_dir, "audio", sample["audio_file"])
        try:
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return a dummy sample
            return {
                "input_ids": torch.zeros(1, dtype=torch.long),
                "attention_mask": torch.zeros(1, dtype=torch.long),
                "audio_values": torch.zeros(1, 1),
                "prompt": "",
                "style_themes": [],
                "instruments": [],
                "tempo": 120,
                "pitch": 0,
                "duration": 10
            }
        
        # Get prompt
        prompt = sample.get("prompt", "")
        
        # Get style/themes
        style_themes = sample.get("style_themes", [])
        
        # Get instruments
        instruments = sample.get("instruments", [])
        
        # Get tempo
        tempo = sample.get("tempo", 120)
        
        # Get pitch
        pitch = sample.get("pitch", 0)
        
        # Get duration
        duration = sample.get("duration", 30)
        
        # Trim audio if too long
        max_samples = int(self.config.max_duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Process audio with MusicGen processor
        try:
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt"
            )
            
            # Process audio separately
            audio_inputs = self.processor(
                audio=audio,
                sampling_rate=sr,
                return_tensors="pt"
            )
            
            # Extract features
            input_ids = inputs["input_ids"][0]
            attention_mask = inputs["attention_mask"][0]
            audio_values = audio_inputs.get("input_values", torch.zeros(1, 1))[0]
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio_values": audio_values,
                "prompt": prompt,
                "style_themes": style_themes,
                "instruments": instruments,
                "tempo": tempo,
                "pitch": pitch,
                "duration": duration
            }
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            # Return a dummy sample
            return {
                "input_ids": torch.zeros(1, dtype=torch.long),
                "attention_mask": torch.zeros(1, dtype=torch.long),
                "audio_values": torch.zeros(1, 1),
                "prompt": "",
                "style_themes": [],
                "instruments": [],
                "tempo": 120,
                "pitch": 0,
                "duration": 10
            }
    
    def collate_fn(self, batch):
        """Collate function for DataLoader"""
        # Collect input_ids and attention_mask
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        
        # Collect audio_values
        audio_values = torch.stack([item["audio_values"] for item in batch])
        
        # Collect other items
        prompts = [item["prompt"] for item in batch]
        style_themes = [item["style_themes"] for item in batch]
        instruments = [item["instruments"] for item in batch]
        tempos = [item["tempo"] for item in batch]
        pitches = [item["pitch"] for item in batch]
        durations = [item["duration"] for item in batch]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_values": audio_values,
            "prompts": prompts,
            "style_themes": style_themes,
            "instruments": instruments,
            "tempos": tempos,
            "pitches": pitches,
            "durations": durations
        }


class MusicGenerator:
    def __init__(self, config):
        self.config = config
        
        # Initialize the MusicGen model and processor
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(config.model_name)
        self.model.to(config.device)
        
        # Initialize style encoder
        self.style_encoder = StyleEncoder(config)
        self.style_encoder.to(config.device)
        
        # Initialize musical attributes encoder
        self.attributes_encoder = MusicalAttributesEncoder(config)
        self.attributes_encoder.to(config.device)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model'])
        self.style_encoder.load_state_dict(checkpoint['style_encoder'])
        self.attributes_encoder.load_state_dict(checkpoint['attributes_encoder'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def save_checkpoint(self, checkpoint_path):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'style_encoder': self.style_encoder.state_dict(),
            'attributes_encoder': self.attributes_encoder.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def _process_audio_reference(self, audio_path, sample_rate=None):
        """Process audio reference for conditioning"""
        if audio_path is None:
            return None
        
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=sample_rate or self.config.sample_rate)
        
        # Process with MusicGen processor
        inputs = self.processor(
            audio=audio, 
            sampling_rate=sr,
            return_tensors="pt"
        ).to(self.config.device)
        
        return inputs
    
    def _process_lyrics(self, lyrics_path=None, lyrics_text=None):
        """Process lyrics for conditioning"""
        if lyrics_path is None and lyrics_text is None:
            return None
        
        # Load lyrics from file if path is provided
        if lyrics_path is not None and os.path.exists(lyrics_path):
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                lyrics_text = f.read()
        
        if lyrics_text is None:
            return None
        
        # Process lyrics with style encoder
        lyrics_embedding = self.style_encoder([lyrics_text])
        
        return lyrics_embedding
    
    def _process_style_themes(self, style_themes, avoid_themes=None):
        """Process style themes for conditioning"""
        if style_themes is None:
            return None
        
        # Convert to list if string
        if isinstance(style_themes, str):
            style_themes = [style_themes]
        
        # Process style themes with style encoder
        style_embedding = self.style_encoder(style_themes)
        
        # Process avoid themes if provided
        if avoid_themes is not None:
            if isinstance(avoid_themes, str):
                avoid_themes = [avoid_themes]
            
            avoid_embedding = self.style_encoder(avoid_themes)
            
            # Subtract avoid embedding from style embedding (simple negative prompting)
            style_embedding = style_embedding - 0.5 * avoid_embedding
        
        return style_embedding
    
    def _process_musical_attributes(self, tempo=None, pitch=None, duration=None, 
                                   instruments=None, avoid_instruments=None):
        """Process musical attributes for conditioning"""
        # Prepare attributes dictionary
        attributes = {}
        
        # Process tempo
        if tempo is not None:
            attributes['tempo'] = torch.tensor([[float(tempo) / 200.0]]).to(self.config.device)
        
        # Process pitch
        if pitch is not None:
            attributes['pitch'] = torch.tensor([[float(pitch) / 12.0]]).to(self.config.device)
        
        # Process duration
        if duration is not None:
            attributes['duration'] = torch.tensor([[min(float(duration), self.config.max_duration) / 
                                                 self.config.max_duration]]).to(self.config.device)
        
        # Process instruments
        if instruments is not None:
            # Simple mapping of instrument names to indices
            # In a real implementation, you'd have a more comprehensive mapping
            instrument_mapping = {
                "piano": 0, "guitar": 1, "drums": 2, "bass": 3, "violin": 4,
                "cello": 5, "flute": 6, "saxophone": 7, "trumpet": 8, "synth": 9
            }
            
            if isinstance(instruments, str):
                instruments = [instruments]
            
            instrument_indices = []
            for instrument in instruments:
                if instrument.lower() in instrument_mapping:
                    instrument_indices.append(instrument_mapping[instrument.lower()])
                else:
                    # Default to piano if instrument not found
                    instrument_indices.append(0)
            
            attributes['instruments'] = torch.tensor(instrument_indices, dtype=torch.long).to(self.config.device)
        
        # Process avoid instruments
        # In a real implementation, you'd use this to modify the generation process
        # For now, we'll just print a message
        if avoid_instruments is not None:
            print(f"Avoiding instruments: {avoid_instruments}")
        
        # Get embeddings from attributes encoder
        attributes_embedding = self.attributes_encoder(attributes)
        
        return attributes_embedding
    
    def _enhance_prompt(self, prompt, style_themes=None, instruments=None, tempo=None, pitch=None):
        """Enhance the text prompt with additional information"""
        enhanced_prompt = prompt
        
        # Add style themes if provided
        if style_themes:
            if isinstance(style_themes, list):
                style_themes_str = ", ".join(style_themes)
            else:
                style_themes_str = style_themes
            enhanced_prompt += f". Style: {style_themes_str}"
        
        # Add instruments if provided
        if instruments:
            if isinstance(instruments, list):
                instruments_str = ", ".join(instruments)
            else:
                instruments_str = instruments
            enhanced_prompt += f". Instruments: {instruments_str}"
        
        # Add tempo if provided
        if tempo:
            enhanced_prompt += f". Tempo: {tempo} BPM"
        
        # Add pitch if provided
        if pitch:
            enhanced_prompt += f". Pitch: {pitch}"
        
        return enhanced_prompt
    
    def generate(self, prompt, output_file=None, audio_reference=None, lyrics=None,
                style_themes=None, avoid_themes=None, tempo=None, pitch=None, 
                duration=30, instruments=None, avoid_instruments=None, 
                vocals_file=None, instrumental_file=None, guidance_scale=None,
                return_audio=False):
        """Generate music with the specified parameters"""
        # Set model to evaluation mode
        self.model.eval()
        self.style_encoder.eval()
        self.attributes_encoder.eval()
        
        # Process audio reference if provided
        audio_inputs = None
        if audio_reference:
            audio_inputs = self._process_audio_reference(audio_reference)
        
        # Process vocals if provided
        vocals_inputs = None
        if vocals_file:
            vocals_inputs = self._process_audio_reference(vocals_file)
        
        # Process instrumental if provided
        instrumental_inputs = None
        if instrumental_file:
            instrumental_inputs = self._process_audio_reference(instrumental_file)
        
        # Enhance prompt with additional information
        enhanced_prompt = self._enhance_prompt(prompt, style_themes, instruments, tempo, pitch)
        
        # Process inputs with MusicGen processor
        inputs = self.processor(
            text=[enhanced_prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.config.device)
        
        # Add audio inputs if provided
        if audio_inputs is not None:
            inputs['audio_values'] = audio_inputs['audio_values']
        
        # Set generation parameters
        generation_kwargs = {
            'max_new_tokens': int(duration * self.config.sample_rate / self.model.config.audio_encoder.hop_length),
            'guidance_scale': guidance_scale or self.config.guidance_scale,
            'conditioning_scale': self.config.conditioning_scale,
        }
        
        # Generate audio
        with torch.no_grad():
            audio_values = self.model.generate(**inputs, **generation_kwargs)
        
        # Convert to numpy array
        audio_array = audio_values.cpu().numpy().squeeze()
        
        # Save audio if output file is provided
        if output_file:
            import soundfile as sf
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            sf.write(output_file, audio_array, self.config.sample_rate)
            print(f"Generated audio saved to {output_file}")
        
        if return_audio:
            return audio_array
        
        return output_file
    
    def train(self, train_dataset, val_dataset=None, num_epochs=None, batch_size=None, 
             learning_rate=None, checkpoint_dir=None, resume_from=None):
        """Train the model on the provided dataset"""
        # Set training parameters
        num_epochs = num_epochs or self.config.max_epochs
        batch_size = batch_size or self.config.batch_size
        learning_rate = learning_rate or self.config.learning_rate
        checkpoint_dir = checkpoint_dir or self.config.checkpoint_dir
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )
        
        # Set models to training mode
        self.model.train()
        self.style_encoder.train()
        self.attributes_encoder.train()
        
        # Create optimizer
        optimizer = AdamW([
            {'params': self.model.parameters()},
            {'params': self.style_encoder.parameters()},
            {'params': self.attributes_encoder.parameters()}
        ], lr=learning_rate, weight_decay=self.config.weight_decay)
        
        # Create learning rate scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs * len(train_loader),
            eta_min=learning_rate / 10
        )
        
        # Resume from checkpoint if provided
        start_epoch = 0
        if resume_from:
            checkpoint = torch.load(resume_from, map_location=self.config.device)
            self.model.load_state_dict(checkpoint['model'])
            self.style_encoder.load_state_dict(checkpoint['style_encoder'])
            self.attributes_encoder.load_state_dict(checkpoint['attributes_encoder'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}")
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            train_loss = 0.0
            
            # Progress bar for training
            progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    audio_values=batch.get('audio_values'),
                    decoder_input_ids=batch.get('decoder_input_ids'),
                    return_dict=True
                )
                
                # Compute loss
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update progress bar
                train_loss += loss.item()
                progress_bar.set_postfix({'loss': train_loss / (progress_bar.n + 1)})
            
            # Validation
            if val_dataset:
                val_loss = 0.0
                
                # Set models to evaluation mode
                self.model.eval()
                self.style_encoder.eval()
                self.attributes_encoder.eval()
                
                with torch.no_grad():
                    for batch in tqdm.tqdm(val_loader, desc="Validation"):
                        # Move batch to device
                        batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                        
                        # Forward pass
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            audio_values=batch.get('audio_values'),
                            decoder_input_ids=batch.get('decoder_input_ids'),
                            return_dict=True
                        )
                        
                        # Compute loss
                        loss = outputs.loss
                        
                        # Update validation loss
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                print(f"Validation loss: {val_loss:.4f}")
                
                # Set models back to training mode
                self.model.train()
                self.style_encoder.train()
                self.attributes_encoder.train()
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'style_encoder': self.style_encoder.state_dict(),
                'attributes_encoder': self.attributes_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


def convert_csv_to_musicgen_dataset(csv_file, output_dir, split_ratio=0.9):
    """Convert CSV dataset to MusicGen training format"""
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    for directory in [
        train_dir, val_dir,
        os.path.join(train_dir, "audio"),
        os.path.join(val_dir, "audio")
    ]:
        os.makedirs(directory, exist_ok=True)
    
    # Read CSV file
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Found {len(rows)} entries in CSV file")
    
    # Process each entry
    train_samples = []
    val_samples = []
    
    for i, row in enumerate(tqdm.tqdm(rows, desc="Processing files")):
        try:
            # Get file path and check if it exists
            audio_path = row["Local_File_Path"]
            if not os.path.exists(audio_path):
                print(f"Warning: File not found: {audio_path}")
                continue
            
            # Load audio to get duration and sample rate
            audio, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
            
            # Skip if duration is too short or too long
            if duration < 5 or duration > 180:  # Adjust thresholds as needed
                print(f"Skipping {audio_path}: duration {duration}s out of range")
                continue
            
            # Extract basic audio features
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            
            # Parse tags into style_themes
            tags = row.get("Tags", "").split(", ") if row.get("Tags") else []
            
            # Create sample metadata
            filename = f"{i:06d}.mp3"
            sample = {
                "audio_file": filename,
                "original_path": audio_path,
                "duration": float(duration),
                "prompt": row.get("Prompt", "") if row.get("Prompt") and row.get("Prompt") != "Prompt not found" else f"A music track with {', '.join(tags) if tags else 'melody'}",
                "style_themes": tags,
                "tempo": float(tempo),
                "pitch": 0,  # Default pitch
                "lyrics": row.get("Lyrics", "") if row.get("Lyrics") and row.get("Lyrics") != "Lyrics not found" else "",
                "name": row.get("Name", "") if row.get("Name") and row.get("Name") != "Title not found" else f"Track {i}",
                "instruments": []  # Unknown instruments
            }
            
            # Decide if sample goes to train or validation set
            if i < int(len(rows) * split_ratio):
                # Copy audio file to train directory
                output_path = os.path.join(train_dir, "audio", filename)
                shutil.copy2(audio_path, output_path)
                train_samples.append(sample)
            else:
                # Copy audio file to validation directory
                output_path = os.path.join(val_dir, "audio", filename)
                shutil.copy2(audio_path, output_path)
                val_samples.append(sample)
        
        except Exception as e:
            print(f"Error processing {row.get('Local_File_Path', 'unknown')}: {e}")
    
    # Save metadata
    train_metadata = {"samples": train_samples}
    val_metadata = {"samples": val_samples}
    
    with open(os.path.join(train_dir, "metadata.json"), "w") as f:
        json.dump(train_metadata, f, indent=2)
    
    with open(os.path.join(val_dir, "metadata.json"), "w") as f:
        json.dump(val_metadata, f, indent=2)
    
    print(f"Prepared {len(train_samples)} training samples and {len(val_samples)} validation samples")
    return train_dir, val_dir


def prepare_dataset(audio_dir, output_dir, split_ratio=0.9):
    """Prepare dataset from raw audio files"""
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    for directory in [
        train_dir, val_dir,
        os.path.join(train_dir, "audio"),
        os.path.join(val_dir, "audio")
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
            
            # Skip if duration is too short or too long
            if duration < 5 or duration > 30:
                continue
            
            # Extract basic audio features for automatic annotation
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
            
            # Simple heuristic to determine style/genre based on audio features
            style = "unknown"
            if tempo < 80:
                style = "slow"
            elif tempo < 120:
                style = "medium"
            else:
                style = "fast"
            
            if spectral_centroid < 1000:
                style += ", bass-heavy"
            elif spectral_centroid < 2000:
                style += ", balanced"
            else:
                style += ", treble-heavy"
            
            # Create sample metadata
            filename = f"{i:06d}.wav"
            sample = {
                "audio_file": filename,
                "original_path": audio_path,
                "duration": duration,
                "prompt": f"A {style} music track",
                "style_themes": [style],
                "tempo": tempo,
                "pitch": 0,  # Default pitch
                "instruments": []  # Unknown instruments
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


def train_from_csv(csv_file, config, num_epochs=10, batch_size=4, learning_rate=1e-5):
    """Train MusicGen model from CSV dataset"""
    print(f"Converting CSV dataset from {csv_file} to MusicGen format...")
    train_dir, val_dir = convert_csv_to_musicgen_dataset(
        csv_file=csv_file, 
        output_dir=config.data_dir,
        split_ratio=0.9
    )
    
    print("Creating datasets...")
    train_dataset = MusicDataset(config, split="train")
    val_dataset = MusicDataset(config, split="val")
    
    print("Creating model...")
    generator = MusicGenerator(config)
    
    print(f"Starting training for {num_epochs} epochs...")
    generator.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_dir=config.checkpoint_dir
    )
    
    return generator


def interactive_generation(generator):
    """Interactive mode for music generation"""
    print("=== Music Generation Interactive Mode ===")
    print("Enter 'q' to quit")
    
    while True:
        # Get prompt
        prompt = input("\nEnter prompt: ")
        if prompt.lower() == 'q':
            break
        
        # Get output file
        output_file = input("Enter output file (default=output/music/generated.wav): ")
        if output_file.lower() == 'q':
            break
        if not output_file:
            output_file = "output/music/generated.wav"
        
        # Get audio reference
        audio_reference = input("Enter audio reference file (optional): ")
        if audio_reference.lower() == 'q':
            break
        if not audio_reference or not os.path.exists(audio_reference):
            audio_reference = None
        
        # Get lyrics
        lyrics = input("Enter lyrics file or text (optional): ")
        if lyrics.lower() == 'q':
            break
        if not lyrics:
            lyrics = None
        elif os.path.exists(lyrics):
            with open(lyrics, 'r', encoding='utf-8') as f:
                lyrics = f.read()
        
        # Get style themes
        style_themes_str = input("Enter style/themes (comma-separated, optional): ")
        if style_themes_str.lower() == 'q':
            break
        style_themes = None
        if style_themes_str:
            style_themes = [theme.strip() for theme in style_themes_str.split(",")]
        
        # Get avoid themes
        avoid_themes_str = input("Enter themes to avoid (comma-separated, optional): ")
        if avoid_themes_str.lower() == 'q':
            break
        avoid_themes = None
        if avoid_themes_str:
            avoid_themes = [theme.strip() for theme in avoid_themes_str.split(",")]
        
        # Get tempo
        tempo_str = input("Enter tempo in BPM (optional): ")
        if tempo_str.lower() == 'q':
            break
        tempo = None
        if tempo_str:
            try:
                tempo = float(tempo_str)
            except ValueError:
                print("Invalid tempo, ignoring")
        
        # Get pitch
        pitch_str = input("Enter pitch adjustment (optional): ")
        if pitch_str.lower() == 'q':
            break
        pitch = None
        if pitch_str:
            try:
                pitch = float(pitch_str)
            except ValueError:
                print("Invalid pitch, ignoring")
        
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
        
        # Get instruments
        instruments_str = input("Enter instruments to include (comma-separated, optional): ")
        if instruments_str.lower() == 'q':
            break
        instruments = None
        if instruments_str:
            instruments = [instrument.strip() for instrument in instruments_str.split(",")]
        
        # Get avoid instruments
        avoid_instruments_str = input("Enter instruments to avoid (comma-separated, optional): ")
        if avoid_instruments_str.lower() == 'q':
            break
        avoid_instruments = None
        if avoid_instruments_str:
            avoid_instruments = [instrument.strip() for instrument in avoid_instruments_str.split(",")]
        
        # Get vocals file
        vocals_file = input("Enter vocals file (optional): ")
        if vocals_file.lower() == 'q':
            break
        if not vocals_file or not os.path.exists(vocals_file):
            vocals_file = None
        
        # Get instrumental file
        instrumental_file = input("Enter instrumental file (optional): ")
        if instrumental_file.lower() == 'q':
            break
        if not instrumental_file or not os.path.exists(instrumental_file):
            instrumental_file = None
        
        # Generate music
        print("\nGenerating music...")
        generator.generate(
            prompt=prompt,
            output_file=output_file,
            audio_reference=audio_reference,
            lyrics=lyrics,
            style_themes=style_themes,
            avoid_themes=avoid_themes,
            tempo=tempo,
            pitch=pitch,
            duration=duration,
            instruments=instruments,
            avoid_instruments=avoid_instruments,
            vocals_file=vocals_file,
            instrumental_file=instrumental_file
        )
        print("Done!")


def main():
    """Main function for training and generating music"""
    parser = argparse.ArgumentParser(description="Music Generation System")
    subparsers = parser.add_subparsers(dest="mode", help="Mode")
    
    # Training parser
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data_dir", type=str, default="data/music", help="Data directory")
    train_parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/musicgen", help="Checkpoint directory")
    train_parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    
    # Generation parser
    gen_parser = subparsers.add_parser("generate", help="Generate music")
    gen_parser.add_argument("--prompt", type=str, required=True, help="Text prompt for music generation")
    gen_parser.add_argument("--output_file", type=str, default="output/music/generated.wav", help="Output audio file")
    gen_parser.add_argument("--audio_reference", type=str, default=None, help="Audio reference file")
    gen_parser.add_argument("--lyrics", type=str, default=None, help="Lyrics file or text")
    gen_parser.add_argument("--style_themes", type=str, default=None, help="Style/themes (comma-separated)")
    gen_parser.add_argument("--avoid_themes", type=str, default=None, help="Themes to avoid (comma-separated)")
    gen_parser.add_argument("--tempo", type=float, default=None, help="Tempo in BPM")
    gen_parser.add_argument("--pitch", type=float, default=None, help="Pitch adjustment")
    gen_parser.add_argument("--duration", type=float, default=30, help="Duration in seconds")
    gen_parser.add_argument("--instruments", type=str, default=None, help="Instruments to include (comma-separated)")
    gen_parser.add_argument("--avoid_instruments", type=str, default=None, help="Instruments to avoid (comma-separated)")
    gen_parser.add_argument("--vocals_file", type=str, default=None, help="Vocals file for integration")
    gen_parser.add_argument("--instrumental_file", type=str, default=None, help="Instrumental file for integration")
    gen_parser.add_argument("--guidance_scale", type=float, default=None, help="Guidance scale for generation")
    gen_parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    
    # Prepare dataset parser
    prep_parser = subparsers.add_parser("prepare", help="Prepare dataset")
    prep_parser.add_argument("--audio_dir", type=str, required=True, help="Audio directory")
    prep_parser.add_argument("--output_dir", type=str, default="data/music", help="Output directory")
    prep_parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/val split ratio")
    
    # CSV training parser
    csv_parser = subparsers.add_parser("train_csv", help="Train from CSV dataset")
    csv_parser.add_argument("--csv_file", type=str, required=True, help="CSV file with dataset")
    csv_parser.add_argument("--data_dir", type=str, default="data/music", help="Data directory")
    csv_parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    csv_parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    csv_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    csv_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/musicgen", help="Checkpoint directory")
    
    # Interactive parser
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    interactive_parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    
    args = parser.parse_args()
    
    # Create config
    config = MusicGenConfig()
    
    if args.mode == "train":
        # Update config with command line arguments
        config.data_dir = args.data_dir
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate
        config.max_epochs = args.epochs
        config.checkpoint_dir = args.checkpoint_dir
        
        # Create datasets
        train_dataset = MusicDataset(config, split="train")
        val_dataset = MusicDataset(config, split="val")
        
        # Create model
        generator = MusicGenerator(config)
        
        # Train model
        generator.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume_from
        )
    
    elif args.mode == "generate":
        # Create model
        generator = MusicGenerator(config)
        
        # Load checkpoint if provided
        if args.checkpoint:
            generator.load_checkpoint(args.checkpoint)
        
        # Parse style themes
        style_themes = None
        if args.style_themes:
            style_themes = [theme.strip() for theme in args.style_themes.split(",")]
        
        # Parse avoid themes
        avoid_themes = None
        if args.avoid_themes:
            avoid_themes = [theme.strip() for theme in args.avoid_themes.split(",")]
        
        # Parse instruments
        instruments = None
        if args.instruments:
            instruments = [instrument.strip() for instrument in args.instruments.split(",")]
        
        # Parse avoid instruments
        avoid_instruments = None
        if args.avoid_instruments:
            avoid_instruments = [instrument.strip() for instrument in args.avoid_instruments.split(",")]
        
        # Generate music
        generator.generate(
            prompt=args.prompt,
            output_file=args.output_file,
            audio_reference=args.audio_reference,
            lyrics=args.lyrics,
            style_themes=style_themes,
            avoid_themes=avoid_themes,
            tempo=args.tempo,
            pitch=args.pitch,
            duration=args.duration,
            instruments=instruments,
            avoid_instruments=avoid_instruments,
            vocals_file=args.vocals_file,
            instrumental_file=args.instrumental_file,
            guidance_scale=args.guidance_scale
        )
    
    elif args.mode == "prepare":
        # Prepare dataset
        prepare_dataset(
            audio_dir=args.audio_dir,
            output_dir=args.output_dir,
            split_ratio=args.split_ratio
        )
    
    elif args.mode == "train_csv":
        # Update config with command line arguments
        config.data_dir = args.data_dir
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate
        config.max_epochs = args.epochs
        config.checkpoint_dir = args.checkpoint_dir
        
        # Train from CSV
        train_from_csv(
            csv_file=args.csv_file,
            config=config,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    
    elif args.mode == "interactive":
        # Create model
        generator = MusicGenerator(config)
        
        # Load checkpoint if provided
        if args.checkpoint:
            generator.load_checkpoint(args.checkpoint)
        
        # Interactive mode
        interactive_generation(generator)


if __name__ == "__main__":
    main()
