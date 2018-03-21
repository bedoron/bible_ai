from bible_ai.bible_loader import BibleLoader, DOWNLOAD_LINK
import keras as k
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model

# Ideas from https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/#comment-432680

def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    """
    :type model: Model
    :type tokenizer: Tokenizer
    :type seq_length: int
    :type seed_text: str
    :type n_words: int
    :rtype: str
    """
    result = list()
    in_text = seed_text
    for _ in range(n_words):
        # Encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # Truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length - 1, truncating='pre')
        # Predict probabilities for each word
        prediction_vector = model.predict(encoded, verbose=0)
        #yhat = (prediction_vector > 0.5).astype('int64')
        #yhat = np.argmax(yhat[0])
        yhat = np.argmax(prediction_vector[0][1:])
        # Map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index != yhat:
                continue

            out_word = word
            break

        # Append to input
        in_text += ' ' + out_word
        result.append(out_word)

    return ' '.join(result)


def run_genenrator():
    bl = BibleLoader(DOWNLOAD_LINK)
    bl.load()

    model = k.models.load_model('model.h5')
    with open('tokenizer.pkl', 'rb') as fd:
        tokenizer = pickle.load(fd)

    print(model.summary())

    seed_text = bl.verses[np.random.randint(0, len(bl.verses))]
    print("Seed text: ", seed_text)
    generated = generate_seq(model, tokenizer, 12, seed_text, 50)
    print("Generated:")
    print("*" * 15)
    print(generated)
    print("*"*15)


if __name__ == "__main__":
    run_genenrator()
