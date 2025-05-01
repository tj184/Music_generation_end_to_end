from flask import Flask, render_template, request
from main import generate_music_by_genre 
from intent_generation import prompt # updated function name if needed

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    user_input = ''
    file_url = ''
    if request.method == 'POST':
        user_input = request.form['music_input']
        intent=prompt(user_input)
        file_path = generate_music_by_genre(intent)  # now returns path to .mp3
        if file_path:
            file_url = '/' + file_path.replace('\\', '/')  # Ensure URL works on Windows
    return render_template('index.html', user_input=user_input, file_url=file_url)

if __name__ == '__main__':
    app.run(debug=True)
