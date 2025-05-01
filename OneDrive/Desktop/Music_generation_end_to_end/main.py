from intent_generation import prompt
from music_generate import generate_music_by_genre

def generate(text):
    # Generate the genre intent from input text
    intent = prompt(text)

    # Generate music and get the file path to the generated MIDI
    midi_file_path = generate_music_by_genre(intent)

    # Return the relative file path to be used in the Flask template
    return midi_file_path
