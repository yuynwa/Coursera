import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer


# 中文 英文 的标点符号不一样， 结果就不一样， 它没区分中文自负
sentences = [
    'I i Love my dog',
    'I Love my cat',
    'You Love my dog！',
]


tokenizer = Tokenizer(num_words=10)
# tokenizer.fit_on_texts(sentences)
tokenizer.fit_on_texts(sentences)

word_idx = tokenizer.word_index
print(word_idx)


