import gensim
from nltk.tokenize import sent_tokenize
import os


training_file = open("../data/blake.txt")
poems1 = open("../data/keats.txt")
poems2 = open("../data/frost_poems.txt")
poems3 = open("../data/byron.txt")
poems4 = open("../data/emily.txt")
poems5 = open("../data/tagore.txt")
#poems6 = open("../data/poems.txt")

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
#corpus7 = poems6.read().lower().split("\n")
#corpus7 = [sentence for sentence in corpus7 if(#sentence != '' and len(sentence) > 1)]
    
corpus = corpus1+corpus2+corpus3+corpus4+corpus5+corpus6
words = [[word for word in sentence.split()] for sentence in corpus]

word_model = gensim.models.Word2Vec(
    words, size=100, min_count=1, window=5, iter=100)


if not os.path.exists('../model'):
    os.makedirs('../model')
word_model.save("../model/word2vec_model")
