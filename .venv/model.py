import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.preprocessing import sequence
from keras.datasets import imdb



vocab = 88584

max_length = 250
batch_size = 64
saved_model = "model.h5"

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab)
# printe = train_data[0]
#
# print(printe)

train_data = sequence.pad_sequences(train_data, max_length)
test_data = sequence.pad_sequences(test_data, max_length)

def build():
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab, 32),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc'])
    return model


    model = build()
    history = model.fit(train_data, train_labels, epochs=5, validation_split=0.2)
    results = model.evaluate(test_data, test_labels)
    print(results)

    model.save(saved_model)
