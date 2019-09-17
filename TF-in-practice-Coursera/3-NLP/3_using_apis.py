import json

with open('./tmp_data/sarcasm.json') as f:
    datastore = json.load(f)



sentences = []
labels = []
urls = []


for x in datastore:
    sentences.append(x['headline'])
    labels.append(x['is_sarcastic'])
    urls.append(x['article_link'])


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)

word_idx = tokenizer.word_index

print(len(word_idx))
print(word_idx)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(sentences[2])
print(padded[2])
print(padded.shape)