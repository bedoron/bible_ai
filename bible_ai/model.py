from tqdm import tqdm

from bible_ai.bible_loader import BibleLoader
from bible_ai.bible_loader import DOWNLOAD_LINK

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import utils as k
from keras.layers import Embedding, Input, LSTM, Dense
from keras.models import Model
import pickle
import numpy as np


# Also needs to install tensorflow

def pre_process_data(verses):
    """
    :type words_dict: list
    :rtype: None
    """
    split_verses = [verse.split() for verse in verses]
    split_verses_lengths = np.array([len(verse_entry) for verse_entry in split_verses])
    averange_verse_length = np.average(split_verses_lengths)
    mean_verse_length = np.asscalar(np.median(split_verses_lengths).astype(dtype=np.int64))
    print("Average verse length: ", averange_verse_length, "Median verse length: ", mean_verse_length)
    print("Median length will be used")

    corpus = [item for sublist in split_verses for item in sublist]
    total_corpus_size = len(corpus)
    text_parts = []

    print("Creating sequences of the length", mean_verse_length)
    for _ in tqdm(range(mean_verse_length, total_corpus_size, mean_verse_length)):
        if len(corpus) == 0:
            break

        corpus_chunk = corpus[:mean_verse_length]
        text_parts.append(corpus_chunk)
        corpus = corpus[mean_verse_length:]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_parts)
    original_sequences = tokenizer.texts_to_sequences(text_parts)

    vocab_size = len(tokenizer.word_index) + 1

    print("Padding sequences")
    aligned_sequneces = []
    for sequence in tqdm(original_sequences):
        aligned_sequence = pad_sequences([sequence], maxlen=mean_verse_length, truncating='post', padding='post')
        aligned_sequneces.append(aligned_sequence[0])

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


def run_and_train_model(verses):
    X, y, seq_length, tokenizer = pre_process_data(verses)
    model = build_model(seq_length, tokenizer, embedding_size=300, lstm_memory_cells=400)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, batch_size=128, epochs=2)

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
    run_and_train_model(bl.verses)
