import tensorflow as tf
from tensorflow import data
from tensorflow import TensorShape
from keras import Sequential
from keras.layers import Embedding, GRU, Dense

import numpy as np

text = open('example.txt', 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))
# print(len(vocab))

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

checkpoint_dir = 'training_checkpoints'

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

model.build(TensorShape([1, None]))

# Some info of model
# model.summary()
text_generated = []

def generation_text(model, start_string):
    num_generations = 300
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    temperature = 0.5
    max_length = 142
    power = 1
    model.reset_states()
    for _ in range(num_generations):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

        if " " in text_generated and (max_length*power-10) < len(text_generated) < (max_length*power+10):
            text_generated.append("\n")
            power += 1

    return (start_string + ''.join(text_generated) + ".")

start_string = input("Start string: ")
# #Generating text
print(generation_text(model, start_string=start_string))
