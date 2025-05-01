import numpy as np
from tensorflow.keras.models import load_model
import pickle
import pretty_midi
import os
import subprocess
from pydub import AudioSegment

SEQUENCE_LENGTH = 50

def generate_music_by_genre(genre_name, model_path="genre_music_lstm.h5",
                             mapping_path="note_mappings.pkl",
                             output_file_prefix="generated"):
    # Load mappings and model
    with open(mapping_path, "rb") as f:
        note_to_int, int_to_note, genre_encoder = pickle.load(f)
    
    model = load_model(model_path)
    
    # Prepare genre vector
    genre_index = genre_encoder.transform([genre_name])[0]
    genre_vector = np.eye(len(genre_encoder.classes_))[genre_index].reshape(1, -1)

    # Create a random seed sequence
    seed_seq = np.random.choice(list(note_to_int.values()), SEQUENCE_LENGTH).tolist()

    # Generate notes
    generated = []
    input_seq = seed_seq[:]
    for _ in range(100):
        input_seq_padded = np.array(input_seq).reshape(1, SEQUENCE_LENGTH)
        prediction = model.predict([input_seq_padded, genre_vector], verbose=0)
        next_note = np.argmax(prediction)
        generated.append(next_note)
        input_seq = input_seq[1:] + [next_note]

    # Convert to MIDI
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    start = 0
    for note_num in generated:
        note_name = int_to_note[note_num]
        pitch = note_name if isinstance(note_name, int) else 60  # fallback to middle C
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + 0.5)
        instrument.notes.append(note)
        start += 0.5
    midi.instruments.append(instrument)
    
    # Ensure 'static' folder exists
    output_dir = "static"
    os.makedirs(output_dir, exist_ok=True)

    # Save MIDI file inside /static
    midi_path = os.path.join(output_dir, f"{output_file_prefix}_{genre_name}.mid")
    midi.write(midi_path)
    print(f"MIDI file generated: {midi_path}")

    # Convert MIDI to WAV using FluidSynth
    wav_path = os.path.join(output_dir, f"{output_file_prefix}_{genre_name}.wav")
    sf2_path = "C:/Users/Lenovo/Downloads/FluidR3_GM/FluidR3_GM.sf2"  # adjust this if needed

    try:
        subprocess.run([
            "fluidsynth",
            "-F", wav_path,
            "-r", "44100",
            sf2_path,
            midi_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        print("Error during MIDI to WAV conversion:", e)
        return None

    # Convert WAV to MP3
    mp3_path = os.path.join(output_dir, f"{output_file_prefix}_{genre_name}.mp3")
    try:
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3")
        print(f"MP3 file created at: {mp3_path}")
    except Exception as e:
        print("Error converting WAV to MP3:", e)
        return None

    # Return MP3 file path for web use
    return mp3_path
