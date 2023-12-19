from nltk.stem import PorterStemmer
import numpy as np
import nltk
nltk.download('punkt')

stemmer = PorterStemmer()


def tokenize_sentence(sentence):
    """
    Tokenizes and stems the input sentence
    Returns: List of stemmed words
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    Stems the input word using the Porter Stemmer
    Returns: Stemmed word
    """
    return stemmer.stem(word.lower())


def create_bag_of_words(tokenized_sentence, words):
    """
    Creates a bag of words representation for the input sentence
    Returns: Bag of words representation
    """
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag
