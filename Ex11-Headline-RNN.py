import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras import utils

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

nyt_dir = 'news-article-categories.csv'
all_headlines = []

try:
    headlines_df = pd.read_csv(nyt_dir, on_bad_lines='skip')
    all_headlines.extend(headlines_df['title'].dropna().tolist())
except Exception as e:
    print(f"Error reading: {e}")

all_headlines = [h for h in all_headlines if h != "Unknown"]

if not all_headlines:
    print("No valid headlines found. Check the CSV file or filtering conditions.")
else:
    print(f"Total headlines collected: {len(all_headlines)}")
    print("Sample headlines:", all_headlines[:20])

    # Tokenize the headlines
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_headlines)
    total_words = len(tokenizer.word_index) + 1
    print('Total words:', total_words)

    # Prepare input sequences
    input_sequences = []
    for line in all_headlines:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            partial_sequence = token_list[:i + 1]
            input_sequences.append(partial_sequence)

    if input_sequences:
        max_sequence_len = max(len(x) for x in input_sequences)
        input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
        
        predictors = input_sequences[:, :-1]
        labels = input_sequences[:, -1]
        
        labels = utils.to_categorical(labels, num_classes=total_words)
        
        model = Sequential()
        model.add(Embedding(total_words, 10, input_length=max_sequence_len - 1))
        model.add(LSTM(100))
        model.add(Dropout(0.1))
        model.add(Dense(total_words, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(predictors, labels, epochs=30, verbose=1)
    else:
        print("No input sequences generated. Ensure your data contains valid headlines.")
