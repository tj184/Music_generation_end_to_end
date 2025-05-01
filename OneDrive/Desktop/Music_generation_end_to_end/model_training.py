import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

data = np.load("music_dataset.npz", allow_pickle=True)
X = data["X"]
genre = data["genre"]
y = data["y"]
vocab_size = int(data["vocab_size"])


# Inputs
note_input = Input(shape=(X.shape[1],), name="note_input")
genre_input = Input(shape=(genre.shape[1],), name="genre_input")

embedding = Embedding(input_dim=vocab_size, output_dim=100)(note_input)
x = LSTM(256, return_sequences=True)(embedding)
x = LSTM(256)(x)

merged = Concatenate()([x, genre_input])
output = Dense(vocab_size, activation='softmax')(merged)

model = Model(inputs=[note_input, genre_input], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
model.summary()

# Train
model.fit([X, genre], y, epochs=30, batch_size=64)
model.save("genre_music_lstm.h5")