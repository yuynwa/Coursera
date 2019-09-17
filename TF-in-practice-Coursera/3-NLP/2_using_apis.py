from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'i love my dog',
    'i love my cat',
    'you love my dog',
    'do you think my dog is amazing ?',
]


tokenizer = Tokenizer(100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)

word_idx = tokenizer.word_index
print('word idx is: ', word_idx)

sequences = tokenizer.texts_to_sequences(sentences)
print('sequences is: ', sequences)

test_data = [
    'i really love my dog',
    'my dog loves my manatee',
]


test_seq = tokenizer.texts_to_sequences(test_data)

print('test seq is: ', test_seq)

padded = pad_sequences(sequences,
                       padding='post',
                       maxlen=5,
                       truncating='post')

print('padded is: \n', padded)