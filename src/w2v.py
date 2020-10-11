import gensim
from nltk.tokenize import sent_tokenize
import os


training_file = open("../data/taylorswift.txt")
poems1 = open("../data/keats.txt")
poems2 = open("../data/frost_poems.txt")
corpus1 = training_file.read().lower().split("\n")
corpus1 = [sentence for sentence in corpus1 if(
    sentence != '' and len(sentence) > 1)]
corpus2 = poems1.read().lower().split("\n")
corpus2 = [sentence for sentence in corpus2 if(
    sentence != '' and len(sentence) > 1)]
corpus3 = poems2.read().lower().split("\n")
corpus3 = [sentence for sentence in corpus3 if(
    sentence != '' and len(sentence) > 1)]
corpus = corpus1+corpus2+corpus3
words = [[word for word in sentence.split()] for sentence in corpus]

word_model = gensim.models.Word2Vec(
    words, size=100, min_count=1, window=5, iter=100)


if not os.path.exists('../model'):
    os.makedirs('../model')
word_model.save("../model/word2vec_model")
