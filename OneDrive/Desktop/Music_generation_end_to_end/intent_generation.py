import ollama


def prompt(prompt):
    desired_model = 'llama3.2:3b'
    ask = "Generate music intent fom the following text , the genre are aggressive_midi,sad_midi,romantic_midi,happy_midi,dramatic_midi ,provide output as only one of the given genre and no other text give output as only one of the five genre, not more than one word just most suitable genre as a single word from those five and lowercase. The text is -"+prompt
    
   
    response = ollama.chat(model=desired_model, messages=[{
        'role': 'user',
        'content': ask,
    }])

    final = response['message']['content']
    return final

