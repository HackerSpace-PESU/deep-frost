# Random-Poem-Generator
A poem generator based on text generation using a Gated Recurral Unit or GRU.

The application generates poems that follow a rhyming scheme (provided by the user). The first sentence of the poem is to be entered by the user as a seed input along with the rhyming scheme. Each sentence of the "poem" is generated independently when seeded with the last word of the sentence.<br>

The last word of the sentence is chosen such that it is similar to the last word of the preceeding sentence and such that it follows the rhyming scheme. It uses Word2Vec word embeddings trained on the model's corpus to infer similar words. If no such word is found it chooses a word in the vocabulary that rhymes with the last word of the preceeding sentence. If no rhyming words are found in the vocabulary, the generator stops executing.

This application drew inspiration from the paper, [Shall I Compare Thee to a Machine-Written Sonnet? An Approach to Algorithmic Sonnet Generation](https://arxiv.org/abs/1811.05067).

## Requirements
* Python 3.x
* tensorflow 2.0
* gensim
* pronouncing
* PyQt5

## How to run Poem Generator
1. Clone the repo 
```bash
git clone https://github.com/Chakita/Random-Poem-Generator-
```
2. To use the pre-trained models for generation, place the models in a folder named models in the same directory as the scripts.
3. To train a new model, use
```bash
   python3 w2v.py
   python3 train_gru.py
```
4. To run the poem generator, use
```bash
   python3 Poem_generator.py
```

## Pre-trained Models
A few pre-trained models can be found [here](https://drive.google.com/drive/folders/1yrmnKJ5h0KfIyt8ZxEiX0Y15kwzUMT9o?usp=sharing). They were trained on a corpus of lyrics by Frost, Taylor Swift and John Keats. They can be found in the data directory.

