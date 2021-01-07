import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Bidirectional, Attention, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from gensim.models import Word2Vec
import os


def reverse_sentence(sentence):
    words = sentence.split()
    words = words[::-1]
    new_sent = " ".join(words)
    return new_sent

training_file = open("../data/blake.txt")
poems1 = open("../data/keats.txt")
poems2 = open("../data/frost_poems.txt")
poems3 = open("../data/byron.txt")
poems4 = open("../data/emily.txt")
poems5 = open("../data/tagore.txt")
corpus1 = training_file.read().lower().split("\n")
corpus1 = [sentence for sentence in corpus1 if(
    sentence != '' and len(sentence) > 1)]
corpus2 = poems1.read().lower().split("\n")
corpus2 = [sentence for sentence in corpus2 if(
    sentence != '' and len(sentence) > 1)]
corpus3 = poems2.read().lower().split("\n")
corpus3 = [sentence for sentence in corpus3 if(
    sentence != '' and len(sentence) > 1)]
corpus4 = poems3.read().lower().split("\n")
corpus4 = [sentence for sentence in corpus4 if(
    sentence != '' and len(sentence) > 1)]
corpus5 = poems4.read().lower().split("\n")
corpus5 = [sentence for sentence in corpus5 if(
    sentence != '' and len(sentence) > 1)]
corpus6 = poems5.read().lower().split("\n")
corpus6 = [sentence for sentence in corpus6 if(
    sentence != '' and len(sentence) > 1)]
corpus = corpus1+corpus2+corpus3+corpus4+corpus5+corpus6

corpus = list(map(reverse_sentence, corpus))
words = [[word for word in sentence.split()] for sentence in corpus]
lengths = [len(sentence) for sentence in corpus]
max_len = max(lengths)
word_model = Word2Vec.load("../model/word2vec_model")
pretrained_weights = word_model.wv.syn0
vocab_size, embedding_size = pretrained_weights.shape


def word2index(word):
    return word_model.wv.vocab[word].index


def index_to_word(index):
    return word_model.wv.index2word[index]


xs = np.zeros([len(words), max_len], dtype=np.int32)
ys = np.zeros([len(words)], dtype=np.int32)
for i, sent in enumerate(words):
    if sent:
        for t, word in enumerate(sent[:-1]):
            xs[i, t] = word2index(word)
        ys[i] = word2index(sent[-1])


model = Sequential()
model.add(Embedding(input_dim=vocab_size,
                    output_dim=embedding_size, weights=[pretrained_weights]))
model.add(GRU(units=embedding_size, return_sequences=True))
model.add(GRU(units=embedding_size))
model.add(Dropout(0.4))
model.add(Dense(units=vocab_size, activation="softmax"))
adam = Adam(lr=0.01)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=adam, metrics=["SparseCategoricalAccuracy"])
history = model.fit(xs, ys, batch_size=128, epochs=100, verbose=1)

if not os.path.exists('../model'):
    os.makedirs('../model')
model.save("../model/poet_gru_model.h5")
