import os
import argparse
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream
from keras.api.models import Sequential, load_model
from keras.api.layers import LSTM, Dense, Dropout, BatchNormalization, Activation
from keras.api.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.api.utils import to_categorical

class MusicGenerator:
    def __init__(self):
        self.notes = []
        self.pitchnames = []
        self.n_vocab = 0
        self.model = None
        self.sequence_length = 100
        self.network_input = None
        self.note_to_int = None
        self.int_to_note = None
        
    def load_data(self, midi_dir):
        """Load and process MIDI files from directory"""
        print(f"Processing MIDI files in {midi_dir}...")
        
        # Check if directory exists
        if not os.path.exists(midi_dir):
            raise FileNotFoundError(f"Directory {midi_dir} does not exist")
            
        midi_files = [f for f in os.listdir(midi_dir) if f.endswith(".mid")]
        
        if not midi_files:
            raise ValueError(f"No MIDI files found in {midi_dir}")
        
        for file in midi_files:
            try:
                midi = converter.parse(os.path.join(midi_dir, file))
                notes_to_parse = midi.flat.notes
                
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        self.notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        self.notes.append('.'.join(str(n) for n in element.normalOrder))
                print(f"Processed {file}")
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
        
        if not self.notes:
            raise ValueError("No notes extracted from MIDI files")
            
        self.pitchnames = sorted(set(self.notes))
        self.n_vocab = len(self.pitchnames)
        
        # Create mappings
        self.note_to_int = dict((note, number) for number, note in enumerate(self.pitchnames))
        self.int_to_note = dict((number, note) for number, note in enumerate(self.pitchnames))
        
        print(f"Loaded {len(self.notes)} notes from {len(midi_files)} files")
        print(f"Vocabulary size: {self.n_vocab} unique notes/chords")
        
        # Save mappings for later use
        self._save_mappings()

    def _save_mappings(self):
        """Save note mappings to file for later use during generation"""
        with open('data/mappings.pkl', 'wb') as f:
            pickle.dump((self.pitchnames, self.note_to_int, self.int_to_note), f)
        print("Saved note mappings to data/mappings.pkl")

    def _load_mappings(self):
        """Load note mappings from file"""
        try:
            with open('data/mappings.pkl', 'rb') as f:
                self.pitchnames, self.note_to_int, self.int_to_note = pickle.load(f)
            self.n_vocab = len(self.pitchnames)
            print(f"Loaded mappings with vocabulary size: {self.n_vocab}")
            return True
        except FileNotFoundError:
            print("No mappings file found. Please train the model first.")
            return False

    def create_sequences(self):
        """Create input/output sequences for training"""
        if not self.notes or not self.note_to_int:
            raise ValueError("No notes or note mappings available. Load data first.")
            
        network_input = []
        network_output = []
        
        for i in range(0, len(self.notes) - self.sequence_length, 1):
            sequence_in = self.notes[i:i + self.sequence_length]
            sequence_out = self.notes[i + self.sequence_length]
            network_input.append([self.note_to_int[char] for char in sequence_in])
            network_output.append(self.note_to_int[sequence_out])

        n_patterns = len(network_input)
        if n_patterns == 0:
            raise ValueError("No patterns created. Check sequence length and data size.")
            
        # Save network_input for later use in generation
        self.network_input = network_input
            
        # Reshape and normalize input
        X = np.reshape(network_input, (n_patterns, self.sequence_length, 1))
        X = X / float(self.n_vocab)
        y = to_categorical(network_output, num_classes=self.n_vocab)
        
        print(f"Created {n_patterns} sequences for training")
        return X, y

    def build_model(self, X):
        """Build LSTM model architecture"""
        print("Building model...")
        
        self.model = Sequential([
            LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
            Dropout(0.3),
            LSTM(512, return_sequences=True),
            Dropout(0.3),
            LSTM(512),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256),
            Activation('relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.n_vocab),
            Activation('softmax')
        ])
        
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model.summary()
        return self.model

    def train(self, X, y, epochs=100, batch_size=64, validation_split=0.2):
        """Train the model with checkpointing"""
        print(f"Training model with {epochs} epochs, batch size {batch_size}...")
        
        # Create directory for model checkpoints if it doesn't exist
        os.makedirs('model', exist_ok=True)
        
        checkpoint = ModelCheckpoint(
            'model/weights-{epoch:02d}-{loss:.4f}.hdf5',
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='loss',
            patience=10,
            verbose=1
        )
        
        # Reduce learning rate when a metric has stopped improving
        reduce_lr = ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=0.0001
        )
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            verbose=1
        )
        
        # Save the final model
        self.model.save('model/final_model.h5')
        print("Model training completed and saved to model/final_model.h5")
        
        return history

    def generate(self, output_length=500, output_file='output.mid', temperature=1.0, seed_sequence=None):
        """Generate new music"""
        print(f"Generating {output_length} notes with temperature {temperature}...")
        
        # Load mappings if not already loaded
        if not self.int_to_note:
            if not self._load_mappings():
                return
        
        # Load model if not already loaded
        if self.model is None:
            try:
                self.model = load_model('model/final_model.h5')
                print("Loaded model from model/final_model.h5")
            except FileNotFoundError:
                # Try to find the latest checkpoint
                try:
                    checkpoint_files = [f for f in os.listdir('model') if f.startswith('weights-') and f.endswith('.hdf5')]
                    if not checkpoint_files:
                        raise FileNotFoundError("No model checkpoints found")
                    
                    latest_checkpoint = sorted(checkpoint_files)[-1]
                    
                    # Create a temporary model to load weights
                    temp_input = np.reshape([0] * self.sequence_length, (1, self.sequence_length, 1))
                    self.build_model(temp_input)
                    
                    self.model.load_weights(os.path.join('model', latest_checkpoint))
                    print(f"Loaded model weights from {latest_checkpoint}")
                except Exception as e:
                    print(f"Error loading model: {str(e)}")
                    return
        
        # Initialize seed sequence
        if seed_sequence is None:
            # Load network_input if not already loaded
            if self.network_input is None:
                try:
                    with open('data/network_input.pkl', 'rb') as f:
                        self.network_input = pickle.load(f)
                except FileNotFoundError:
                    print("No network input data found. Using random seed.")
                    # Create a random seed
                    self.network_input = [[np.random.randint(0, self.n_vocab) for _ in range(self.sequence_length)]]
            
            # Select a random starting sequence
            start = np.random.randint(0, len(self.network_input)-1)
            pattern = self.network_input[start]
        else:
            # Use provided seed sequence
            pattern = seed_sequence
        
        # Convert pattern to list if it's a numpy array
        pattern = list(pattern)
            
        # Generate notes
        prediction_output = []
        for _ in range(output_length):
            # Reshape and normalize
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(self.n_vocab)
            
            # Generate prediction
            prediction = self.model.predict(prediction_input, verbose=0)[0]
            
            # Apply temperature for randomness
            if temperature != 1.0:
                prediction = np.log(prediction) / temperature
                prediction = np.exp(prediction) / np.sum(np.exp(prediction))
            
            # Sample from the prediction
            if temperature < 0.5:
                # For low temperatures, just take the argmax
                index = np.argmax(prediction)
            else:
                # For higher temperatures, sample from the distribution
                index = np.random.choice(len(prediction), p=prediction)
            
            result = self.int_to_note[index]
            prediction_output.append(result)
            
            # Update pattern for next prediction
            pattern.append(index)
            pattern = pattern[1:]

        # Create MIDI file
        self.create_midi(prediction_output, output_file)
        return prediction_output

    def create_midi(self, prediction_output, filename):
        """Convert generated notes to MIDI file"""
        print(f"Creating MIDI file {filename}...")
        
        offset = 0
        output_notes = []
        
        for pattern in prediction_output:
            # Check if the pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                try:
                    notes_in_chord = pattern.split('.')
                    notes = []
                    for current_note in notes_in_chord:
                        new_note = note.Note(int(current_note))
                        new_note.storedInstrument = instrument.Piano()
                        notes.append(new_note)
                    new_chord = chord.Chord(notes)
                    new_chord.offset = offset
                    output_notes.append(new_chord)
                except Exception as e:
                    print(f"Error creating chord from {pattern}: {str(e)}")
                    continue
            else:
                # Pattern is a note
                try:
                    new_note = note.Note(pattern)
                    new_note.offset = offset
                    new_note.storedInstrument = instrument.Piano()
                    output_notes.append(new_note)
                except Exception as e:
                    print(f"Error creating note from {pattern}: {str(e)}")
                    continue
            
            # Move forward in time
            offset += 0.5

        # Create a music stream
        midi_stream = stream.Stream(output_notes)
        
        # Create directory for output if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Write the MIDI file
        try:
            midi_stream.write('midi', fp=filename)
            print(f"Generated MIDI file saved as {filename}")
        except Exception as e:
            print(f"Error writing MIDI file: {str(e)}")

def main():
    # Create directories if they don't exist
    os.makedirs('model', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    parser = argparse.ArgumentParser(description='Music Generation with LSTM')
    parser.add_argument('--train', action='store_true', help='Train new model')
    parser.add_argument('--midi_dir', type=str, default='midi_files', 
                       help='Directory containing MIDI files')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--sequence_length', type=int, default=100,
                       help='Length of input sequences')
    parser.add_argument('--generate', action='store_true',
                       help='Generate new music')
    parser.add_argument('--output_length', type=int, default=500,
                       help='Number of notes to generate')
    parser.add_argument('--output_file', type=str, default='output/generated.mid',
                       help='Output MIDI filename')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Randomness of predictions (higher = more random)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    args = parser.parse_args()

    generator = MusicGenerator()
    
    # Set sequence length
    generator.sequence_length = args.sequence_length
    
    if args.interactive:
        interactive_mode(generator)
    else:
        if args.train:
            # Training workflow
            try:
                generator.load_data(args.midi_dir)
                X, y = generator.create_sequences()
                
                # Save network_input for later use in generation
                with open('data/network_input.pkl', 'wb') as f:
                    pickle.dump(generator.network_input, f)
                
                model = generator.build_model(X)
                generator.train(X, y, epochs=args.epochs, batch_size=args.batch_size)
            except Exception as e:
                print(f"Error during training: {str(e)}")
            
        if args.generate:
            # Generation workflow
            try:
                generator.generate(
                    output_length=args.output_length,
                    output_file=args.output_file,
                    temperature=args.temperature
                )
            except Exception as e:
                print(f"Error during generation: {str(e)}")

def interactive_mode(generator):
    """Run the generator in interactive mode with user input"""
    print("=== Interactive Music Generator ===")
    print("1. Train a new model")
    print("2. Generate music")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        midi_dir = input("Enter directory containing MIDI files: ")
        epochs = int(input("Enter number of epochs (default 100): ") or "100")
        batch_size = int(input("Enter batch size (default 64): ") or "64")
        
        try:
            generator.load_data(midi_dir)
            X, y = generator.create_sequences()
            
            # Save network_input for later use in generation
            with open('data/network_input.pkl', 'wb') as f:
                pickle.dump(generator.network_input, f)
            
            model = generator.build_model(X)
            generator.train(X, y, epochs=epochs, batch_size=batch_size)
        except Exception as e:
            print(f"Error during training: {str(e)}")
        
        # Return to menu
        interactive_mode(generator)
        
    elif choice == '2':
        output_length = int(input("Enter number of notes to generate (default 500): ") or "500")
        output_file = input("Enter output filename (default output/generated.mid): ") or "output/generated.mid"
        temperature = float(input("Enter temperature (0.1-2.0, default 1.0): ") or "1.0")
        
        try:
            generator.generate(
                output_length=output_length,
                output_file=output_file,
                temperature=temperature
            )
        except Exception as e:
            print(f"Error during generation: {str(e)}")
        
        # Return to menu
        interactive_mode(generator)
        
    elif choice == '3':
        print("Exiting...")
    else:
        print("Invalid choice. Please try again.")
        interactive_mode(generator)

if __name__ == "__main__":
    main()
