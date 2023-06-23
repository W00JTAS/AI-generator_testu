import os

import keras.callbacks
from tensorflow import data
from keras import Sequential
from keras.layers import Embedding, GRU, Dense
from keras import losses
from keras.callbacks import ModelCheckpoint

import numpy as np

text = open('example.txt', 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))
print(len(vocab))

char2idx = {unique: idx for idx, unique in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[char] for char in text])

seq_length = 100
char_dataset = data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

batch_size = 64
buffer_size = 35162

ds = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    return Sequential(
        [
            Embedding(
                vocab_size,
                embedding_dim,
                batch_input_shape=[batch_size, None],
            ),
            GRU(
                rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform',
                ),
                Dense(vocab_size),
            ]
        )

# Model building
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)

for input_example_batch, target_example_batch in ds.take(1):
   examples_batch_predictions = model(input_example_batch)
   print(examples_batch_predictions.shape)

model.summary()

def loss(labels, logits):
    return losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# Compiling model
model.compile(optimizer='adam', loss=loss)

checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'chkpt_{epoch}')
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 200

# Making checkpoints for training
history = model.fit(ds, epochs=EPOCHS, callbacks=[checkpoint_callback])
