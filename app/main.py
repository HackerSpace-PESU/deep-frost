from tensorflow import keras
from gensim.models import Word2Vec
import numpy as np
import pronouncing
import random
import os
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import InputRequired, Length, Email, EqualTo, ValidationError


sess = tf.compat.v1.Session()
set_session(sess)
model = keras.models.load_model("/app/model/poet_gru_model.h5",compile=False)
word_model = Word2Vec.load("/app/model/word2vec_model")
word_vec = word_model.wv


def word2index(word):
    return word_model.wv.vocab[word].index


def index_to_word(index):
    return word_model.wv.index2word[index]


def reverse_sentence(sentence):
    words = sentence.split()
    words = words[::-1]
    new_sent = " ".join(words)
    return new_sent


def temp_sample(preds, temp=1.0):
    if temp <= 0:
        return np.argmax(preds)
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds)/temp
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    prob = np.random.multinomial(1, preds, 1)
    return np.argmax(prob)


def generate_sent(text, num_generated=5):
    word_indices = [word2index(word) for word in text.lower().split()]
    for i in range(num_generated):
        prediction = model.predict(x=np.array(word_indices))
        index = temp_sample(prediction[-1], 0.7)
        word_indices.append(index)
    return " ".join(index_to_word(index) for index in word_indices)


def generatePoem(scheme, starting):
    possible = []
    first_end_word = starting.split()[-1]
    rhyming_words = pronouncing.rhymes(first_end_word)
    try:
    	most_sim = list(word_model.most_similar(first_end_word))
    except:
    	return "Word not found in vocabulary"
    for word, prob in most_sim:
        if word in rhyming_words:
            possible.append(word)
    if not possible:
        for word in rhyming_words:
            if word in word_vec.vocab:
                possible.append(word)
    if not possible:
        return "No rhyming words found! :("

    scheme_dictionary = {}
    scheme_dictionary[scheme[0]] = possible
    new_words_list = []
    scheme = scheme[1:]
    for letter in scheme:
        if letter not in scheme_dictionary:
            new_word, prob = random.choice(most_sim)
            rhyming_words = pronouncing.rhymes(new_word)
            possible = [
                word for word in word_vec.vocab if word in rhyming_words]
            while len(possible) < 1:
                new_word = random.choice(list(word_vec.vocab.keys()))
                rhyming_words = pronouncing.rhymes(new_word)
                possible = [
                    word for word in word_vec.vocab if word in rhyming_words]
            new_words_list.append(new_word)
            scheme_dictionary[letter] = possible

    if not possible:
        return "No rhyming words found! :("

    poem = starting+"\n"
    for letter in scheme:
        word = random.choice(scheme_dictionary.get(letter))
        rev_sent = generate_sent(word)
        sent = reverse_sentence(rev_sent)
        poem += sent+"\n"
    return poem


app = Flask(__name__)
app.config['SECRET_KEY'] = 'thisisasecret'


class PoemForm(FlaskForm):
    scheme = StringField('Rhyme Scheme:',
                         validators = [
                             InputRequired(message='Field is required!'),
                             Length(min=4, message='scheme should be minimum of length 4')
                             ])
    starting = StringField('First Line:',
                           validators = [
                               InputRequired(message = 'Field is required!'),
                               Length(min=15, message='Input is too small')
                           ])
    
    def __init__(self, *args, **kwargs):
        super(PoemForm, self).__init__(*args, **kwargs)


@app.route('/', methods=['GET', 'POST'])
def home():
    form = PoemForm()
    poem=0
    if form.validate_on_submit():
        scheme = form.scheme.data
        starting = form.starting.data
        poem = generatePoem(scheme, starting).split('\n')
    return render_template('index.html', form = form, poem = poem)


if __name__ == '__main__':
    app.run(debug=True)
