# =============================================================================
# Automatisierter Text aus einer Vorlage
# =============================================================================

import tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import random
import sys
import re
import logging
import os

safeweights = r"\weights"

raw_text = ""

title = "bible.txt"
with open(title, "r")as file:
    raw_textinput = file.read().replace('\n', '')
raw_text = raw_text + raw_textinput
raw_text = raw_text.replace('. . .', '...')
raw_text = raw_text.replace(' - ', ' ')

# Text wird umformatiert, um die Anzahl von unique characters zu reduzieren.
# Dies ist notwendig, um die Anzahl Dummy-Variabeln überschauber zu halten.
processed_text = raw_text.lower()
processed_text = re.sub(r'[^\x00-\x7f]', r'', processed_text)

print('corpus length:', len(processed_text))

chars = sorted(list(set(processed_text)))
print('total chars:', len(chars))


# Erzeugen der Character-list
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters --
# this step generates a sequence of 40 characters in steps of 3.
# hereby the entire text can be split in these sequences.
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(processed_text) - maxlen, step):
    sentences.append(processed_text[i: i + maxlen])
    next_chars.append(processed_text[i + maxlen])
print('nb sequences:', len(sentences))

# Dummy Variabeln werden erzeugt und x / y werden vektorisiert:
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# Das Model wird als LSTM aufgebaut, welches die TF-Dokumentation empfiehlt

print('Build model...')
model = Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.summary()

# Text wird mit Sample-Funktion erzeugt:
# softmax funktion, um den Output in eine Wahrscheinlichkeit umzuwandeln


def sample(preds, temperature):
    # Hilfsfunktion um ein Sample zu ziehen
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Am Ende jeder Epoche soll der Fortschritt dargestellt werden, indem ein
# Textschnipsel ausgegeben wird.


def on_epoch_end(epoch, _):
    print("******************************************************")
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(processed_text) - maxlen - 1)
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('----- temperature:', temperature)

        generated = ''
        sentence = processed_text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


# warnungen von TF 2 werden ignoriert (sind hier unrelevant)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Fit the model
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=30,
          callbacks=[print_callback]
          )

# model.save_weights(filepath=safeweights)