import os
from basic_pitch.inference import predict
from pydub import AudioSegment
import pretty_midi

def convert_to_wav(input_path, output_path):
    if input_path.endswith('.mp3'):
        audio = AudioSegment.from_mp3(input_path)
        audio.export(output_path, format='wav')
        return output_path
    return input_path

def convert_audio_to_midi(audio_path, output_midi_path):
    try:
        model_output, midi_data, note_events = predict(audio_path)
        if midi_data:
            midi_data.write(output_midi_path)
            print(f"Saved MIDI file to {output_midi_path}")
        else:
            print(f"Failed to generate MIDI data for {audio_path}")
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

def process_music_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file in os.listdir(folder_path):
        if file.endswith('.mp3') or file.endswith('.wav'):
            audio_path = os.path.join(folder_path, file)
            wav_path = convert_to_wav(audio_path, audio_path.replace('.mp3', '.wav'))
            midi_output_path = os.path.join(output_folder, file.replace('.mp3', '.mid').replace('.wav', '.mid'))
            convert_audio_to_midi(wav_path, midi_output_path)
            print(f'Converted {file} to MIDI')

if __name__ == "__main__":
    input_music_folder = "mp3"
    output_midi_folder = "output_midi"
    process_music_folder(input_music_folder, output_midi_folder)