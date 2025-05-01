import os
import numpy as np
import pretty_midi
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

DATASET_PATH = 'Dataset_midi'
SEQUENCE_LENGTH = 50  # Number of time steps

def midi_to_note_sequence(midi_path):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    notes.append(note.pitch)
        return sorted(notes)
    except Exception as e:
        print(f"Error with {midi_path}: {e}")
        return []

def process_dataset():
    all_notes = []
    genres = []
    genre_note_sequences = {}

    for genre_folder in os.listdir(DATASET_PATH):
        genre_path = os.path.join(DATASET_PATH, genre_folder)
        if not os.path.isdir(genre_path):
            continue
        genre_notes = []
        for file in os.listdir(genre_path):
            if file.endswith('.mid') or file.endswith('.midi'):
                midi_path = os.path.join(genre_path, file)
                notes = midi_to_note_sequence(midi_path)
                if notes:
                    genre_notes.extend(notes)
                    genres.append(genre_folder)
        genre_note_sequences[genre_folder] = genre_notes
        all_notes.extend(genre_notes)

    # Create a unique note-to-int mapping
    unique_notes = sorted(set(all_notes))
    note_to_int = {note: number for number, note in enumerate(unique_notes)}
    int_to_note = {number: note for note, number in note_to_int.items()}

    X, y, genre_labels = [], [], []

    genre_encoder = LabelEncoder()
    genre_list = list(genre_note_sequences.keys())
    genre_encoder.fit(genre_list)

    for genre, notes in genre_note_sequences.items():
        encoded_notes = [note_to_int[note] for note in notes]
        for i in range(len(encoded_notes) - SEQUENCE_LENGTH):
            X.append(encoded_notes[i:i+SEQUENCE_LENGTH])
            y.append(encoded_notes[i+SEQUENCE_LENGTH])
            genre_labels.append(genre)

    X = np.array(X)
    y = to_categorical(y, num_classes=len(unique_notes))
    genre_encoded = genre_encoder.transform(genre_labels)
    genre_encoded = to_categorical(genre_encoded)

    # Save mappings
    with open('note_mappings.pkl', 'wb') as f:
        pickle.dump((note_to_int, int_to_note, genre_encoder), f)

    return X, genre_encoded, y, len(unique_notes)

if __name__ == "__main__":
    X, genre_labels, y, vocab_size = process_dataset()
    np.savez("music_dataset.npz", X=X, genre=genre_labels, y=y, vocab_size=vocab_size)
