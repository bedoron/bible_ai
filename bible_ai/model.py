from bible_ai.bible_loader import BibleLoader
from bible_ai.bible_loader import DOWNLOAD_LINK

from keras.preprocessing.text import Tokenizer
from keras import utils as k
from keras.layers import Embedding, Input, LSTM, Dense
from keras.models import Model
import pickle
import numpy as np


# Also needs to install tensorflow

def pre_process_data(verses, max_len):
    """
    :type words_dict: list
    :rtype: None
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(verses)
    original_sequences = tokenizer.texts_to_sequences(verses)

    vocab_size = len(tokenizer.word_index) + 1

    aligned_sequneces = []
    for sequence in original_sequences:
        aligned_sequence = np.zeros(max_len, dtype=np.int64)
        aligned_sequence[:len(sequence)] = np.array(sequence, dtype=np.int64)
        aligned_sequneces.append(aligned_sequence)

    sequences = np.array(aligned_sequneces)

    X, y = sequences[:, :-1], sequences[:, -1]
    y = k.to_categorical(y, num_classes=vocab_size)

    seq_lengh = X.shape[1]
    return X, y, seq_lengh, tokenizer


def build_model(input_length, tokenizer, embedding_size = 50, lstm_memory_cells = 100):
    """
    :type input_length: int
    :type tokenizer: Tokenizer
    :return:
    """

    input = Input(shape=(input_length, ))
    X = input

    vocab_size = len(tokenizer.word_counts) + 1

    X = Embedding(vocab_size, embedding_size, input_length=input_length)(X)
    X = LSTM(lstm_memory_cells, return_sequences=True)(X)
    X = LSTM(lstm_memory_cells)(X)
    X = Dense(lstm_memory_cells, activation='relu')(X)
    X = Dense(vocab_size, activation='softmax')(X)

    return Model(inputs=input, outputs=X)


def run_and_train_model(verses, maximal_verse):
    X, y, seq_length, tokenizer = pre_process_data(verses, maximal_verse)
    model = build_model(seq_length, tokenizer, embedding_size=300, lstm_memory_cells=400)
    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, batch_size=128, epochs=1)

    store_model(model, tokenizer)


def store_model(model, tokenizer):
    model.save('model.h5')
    with open('tokenizer.pkl', 'wb') as fd:
        pickle.dump(tokenizer, fd)


if __name__ == "__main__":
    bl = BibleLoader(DOWNLOAD_LINK)
    bl.load()

    print('Unique words: ', len(bl.bible_words.keys()))
    print('Maximal verse length: ', bl.maximal_verse)
    print('Total verses: ', len(bl.verses))
    print('**************************************')
    print()
    run_and_train_model(bl.verses, bl.maximal_verse)
