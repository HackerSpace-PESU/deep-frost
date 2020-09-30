# The poem generator which generates sentences according to the rhyming schemes and then combines these sentences into a "poem".
# Each sentence in the "poem" is generated independently, the last words are chosen according to a rhyming scheme provided by the user.
# Words that rhyme with and are most similar to the last word in the preceeding sentence are picked as seed for the current sentence.
# Rhyming scheme --------> input format="AABBCCDD"---> means 8 lines, every two lines rhyming.
# Check if previous character is the same as current. If it is, continue with words rhyming with it, else choose a new word and proceed.

from PyQt5 import QtCore, QtGui, QtWidgets
from tensorflow import keras
from gensim.models import Word2Vec
import numpy as np
import pronouncing
import random
import sys
import os


model = keras.models.load_model("models/poet_gru_model")
word_model = Word2Vec.load("models/word2vec_model")
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
    most_sim = list(word_model.most_similar(first_end_word))
    for word, prob in most_sim:
        if word in rhyming_words:
            possible.append(word)
    if not possible:
        for word in rhyming_words:
            if word in word_vec.vocab:
                possible.append(word)
    if not possible:
        print("No rhyming words found! :(")
        sys.exit()

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
        print("No rhyming words found! :(")
        sys.exit()

    poem = starting+"\n"
    for letter in scheme:
        word = random.choice(scheme_dictionary.get(letter))
        rev_sent = generate_sent(word)
        sent = reverse_sentence(rev_sent)
        poem += sent+"\n"
    dir_name = "Generated_poems"
    root_dir = "/home"
    if(os.path.isdir(dir_name)):
        os.chdir(dir_name)
        with open("poem.txt", "w") as f:
            f.write(poem)
    else:
        os.mkdir(dir_name)
        os.chdir(dir_name)
        with open("poem.txt", "w") as f:
            f.write(poem)
    os.chdir(root_dir)
    return poem


class Ui_MainWindow(object):
    def clicked(self, text):
        self.label_3.setText(text)
        self.label_3.adjustSize()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1001, 692)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.first_line = QtWidgets.QLineEdit(self.centralwidget)
        self.first_line.setGeometry(QtCore.QRect(120, 70, 361, 25))
        self.first_line.setObjectName("first_line")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(500, 70, 80, 25))
        self.pushButton.setObjectName("pushButton")
        starting = self.pushButton.clicked.connect(self.line_input)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 80, 98, 17))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 151, 17))
        self.label_2.setObjectName("label_2")
        self.rhyming_scheme = QtWidgets.QLineEdit(self.centralwidget)
        self.rhyming_scheme.setGeometry(QtCore.QRect(140, 10, 131, 25))
        self.rhyming_scheme.setObjectName("rhyming_scheme")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(290, 10, 89, 25))
        self.pushButton_2.setObjectName("pushButton_2")
        scheme = self.pushButton_2.clicked.connect(self.rhyme_input)
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(460, 10, 251, 25))
        self.comboBox.setEditable(False)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 110, 831, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 140, 871, 441))
        font = QtGui.QFont()
        font.setPointSize(24)
        font.setItalic(True)
        self.label_3.setFont(font)
        self.label_3.setWordWrap(False)
        self.label_3.setIndent(5)
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1001, 22))
        self.menubar.setObjectName("menubar")
        self.menuGenerate = QtWidgets.QMenu(self.menubar)
        self.menuGenerate.setObjectName("menuGenerate")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionGenerate_new = QtWidgets.QAction(MainWindow)
        self.actionGenerate_new.setObjectName("actionGenerate_new")
        self.menuGenerate.addAction(self.actionGenerate_new)
        self.menubar.addAction(self.menuGenerate.menuAction())
        self.actionGenerate_new.triggered.connect(
            lambda: self.clicked(generatePoem(self.scheme, self.starting)))
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Frost"))
        self.pushButton.setText(_translate("MainWindow", "OK"))
        self.label.setText(_translate("MainWindow", "Enter first line"))
        self.label_2.setText(_translate("MainWindow", "Rhyming Scheme"))
        self.pushButton_2.setText(_translate("MainWindow", "OK"))
        self.comboBox.setCurrentText(_translate(
            "MainWindow", "Robert Frost + Taylor Swift +John Keats"))
        self.comboBox.setItemText(0, _translate(
            "MainWindow", "Robert Frost + Taylor Swift + John Keats"))
        self.label_3.setText(_translate(
            "MainWindow", "Click Generate or ctrl+G to generate a new random poem"))
        self.menuGenerate.setTitle(_translate("MainWindow", "Generate"))
        self.actionGenerate_new.setText(
            _translate("MainWindow", "Generate new"))
        self.actionGenerate_new.setStatusTip(
            _translate("MainWindow", "Generate a new poem"))
        self.actionGenerate_new.setShortcut(_translate("MainWindow", "Ctrl+G"))

    def rhyme_input(self):
        rs = str(self.rhyming_scheme.text())
        self.scheme = rs

    def line_input(self):
        fs = str(self.first_line.text())
        self.starting = fs


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
