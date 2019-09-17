import tensorflow as tf
import numpy as np

print(np.__version__)
print(tf.__version__)

# call this func if version is 1.x
tf.enable_eager_execution()

import tensorflow_datasets as tfds

imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

print(imdb)
print(info)


train_data, test_data = imdb['train'], imdb['test']


training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

z = {'s':'d'}

for s, l in train_data:
    training_sentences.append(s.numpy())
    training_labels.append(l.numpy())


for s, l in test_data:
    testing_sentences.append(s.numpy())
    testing_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)

padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)


testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sentences, maxlen=max_length)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),

    # or alternatively , more faster, less accuracy
    # tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])




