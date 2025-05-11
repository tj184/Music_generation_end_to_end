import numpy as np
from tensorflow.keras.models import load_model
import pickle
import pretty_midi
import os
import subprocess
from pydub import AudioSegment
from collections import Counter

SEQUENCE_LENGTH = 50

def generate_music_by_genre(genre_name, model_path="genre_music_lstm.h5",
                             mapping_path="note_mappings.pkl",
                             output_file_prefix="generated"):

    # Define your custom genre-instrument settings with genre-specific drums
    genre_instruments = {
        "sad_midi": {
            "program": 0,  # Acoustic Grand Piano
            "chords": [[60, 63, 67], [57, 60, 64], [55, 59, 62]],  # Minor chords
            "drum_pattern": [36, 38],  # Kick, Snare (lighter drums)
            "additional_instruments": [41]  # French Horn
        },
        "romance_midi": {
            "program": 40,  # Violin
            "chords": [[60, 64, 67], [62, 65, 69], [59, 63, 67]],  # Major/7th chords
            "drum_pattern": [36, 38, 42],  # Kick, Snare, Hi-hat
            "additional_instruments": [72]  # String Ensemble
        },
        "dramatic_midi": {
            "program": 19,  # Church Organ
            "chords": [[60, 64, 67, 70], [62, 65, 69, 72]],  # Tension chords
            "drum_pattern": [36, 41],  # Kick, Bass Drum (heavier for drama)
            "additional_instruments": [56]  # Timpani
        },
        "aggressive_midi": {
            "program": 30,  # Overdriven Guitar
            "chords": [[48, 52, 55], [47, 50, 54], [45, 49, 52]],
            "drum_pattern": [36, 38, 41, 46],  # Kick, Snare, Bass Drum, Crash Cymbal
            "additional_instruments": [27]  # Distorted Guitar
        },
        "happy_midi": {
            "program": 6,  # Harpsichord
            "chords": [[60, 64, 67], [62, 65, 69], [64, 67, 71]],
            "drum_pattern": [],  # Removed drums for happy_midi
            "additional_instruments": [9]  # Flute
        }
    }

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
    melody_instrument = pretty_midi.Instrument(program=0)
    start = 0
    for note_num in generated:
        note_name = int_to_note[note_num]
        pitch = note_name if isinstance(note_name, int) else 60  # fallback
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + 0.5)
        melody_instrument.notes.append(note)
        start += 0.5
    midi.instruments.append(melody_instrument)

    output_dir = "static"
    os.makedirs(output_dir, exist_ok=True)
    midi_path = os.path.join(output_dir, f"{output_file_prefix}_{genre_name}.mid")
    midi.write(midi_path)

    # Enhance with harmony
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    total_duration = midi_data.get_end_time()

    # Estimate key root from melody
    melody_pitches = [note.pitch % 12 for inst in midi_data.instruments if not inst.is_drum for note in inst.notes]
    key_root = Counter(melody_pitches).most_common(1)[0][0] if melody_pitches else 0

    # Genre-specific settings
    genre_settings = genre_instruments.get(genre_name.lower(), genre_instruments["happy_midi"])
    chord_list = genre_settings["chords"]
    instrument_program = genre_settings["program"]
    drum_pattern = genre_settings["drum_pattern"]
    additional_instruments = genre_settings["additional_instruments"]

    # Add Harmony Instrument
    harmony = pretty_midi.Instrument(program=instrument_program)
    chord_length = 2.0
    t = 0.0
    while t < total_duration:
        chord = chord_list[int((t // chord_length) % len(chord_list))]
        for note in chord:
            harmony.notes.append(pretty_midi.Note(velocity=80, pitch=note, start=t, end=min(t + chord_length, total_duration)))
        t += chord_length
    midi_data.instruments.append(harmony)

    # Add Drums using genre-specific patterns, excluding "happy_midi"
    if drum_pattern and genre_name.lower() != "happy_midi":
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        t = 0.0
        while t < total_duration:
            for drum in drum_pattern:
                drums.notes.append(pretty_midi.Note(velocity=100, pitch=drum, start=t, end=t + 0.1))
            t += 0.5
        midi_data.instruments.append(drums)

    # Add additional instruments if specified
    for instrument_program in additional_instruments:
        additional_instrument = pretty_midi.Instrument(program=instrument_program)
        t = 0.0
        while t < total_duration:
            additional_instrument.notes.append(pretty_midi.Note(velocity=70, pitch=60, start=t, end=t + 0.5))
            t += 0.5
        midi_data.instruments.append(additional_instrument)

    # Save enhanced MIDI
    midi_data.write(midi_path)
    print(f"MIDI file generated: {midi_path}")

    # Convert to WAV with FluidSynth
    wav_path = os.path.join(output_dir, f"{output_file_prefix}_{genre_name}.wav")
    sf2_path = "C:/Users/Lenovo/Downloads/FluidR3_GM/FluidR3_GM.sf2"
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

    return mp3_path
