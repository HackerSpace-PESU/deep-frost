 **Random-Poem-Generator-
A poem generator based on text generation using GRU.**

Generates poems that follow a rhyming scheme(provided by the user).
The first sentence of the poem is to be entered by the user along with the rhyming scheme.
Each sentence of the "poem" is generated independently when seeded with the last word of the sentence.
The last word of the sentence is chosen such that it is similar to the last word of the preceeding sentence and such that it follows the rhyming scheme.
If no such word is found it chooses a word in the vocabulary that rhymes with the last word of the preceeding sentence.
If no rhyming words are found in the vocabulary, the generator stops executing.

**Requirements**:<BR>
Python 3.x<BR>
Tensorflow 2.0<BR>
Gensim<BR>
Pronouncing<BR>
PyQt5<BR>

**To run locally:**
1) Clone the repo 
2) To use the pre-trained models for generation, place the models in a folder named models in the same directory.
3) To train new models:<br>
   python3 w2v.py<br> 
   python3 train_gru.py
4) To run the poem generator:<br>
   python3 Poem_generator.py


**Pre-trained models-**
https://drive.google.com/drive/folders/1yrmnKJ5h0KfIyt8ZxEiX0Y15kwzUMT9o?usp=sharing
