import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


class Bot:

    def __init__(self):
        nltk.download('punkt')

    def display(self):
        return "Hello World"

    def tokenize_sentence(self, sentence):
        return nltk.word_tokenize(sentence)

    def stem(self, word):
        pass


    def bag_of_words(self, tokenized_sentence, all_words):
        pass
