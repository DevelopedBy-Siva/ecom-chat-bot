import json
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
        return stemmer.stem(word.lower())


    def bag_of_words(self, tokenized_sentence, all_words):
        pass

    def prepare_training_data(self):
        with open("data/intents.json", 'r') as intents_file:
            intents_data = json.load(intents_file)

        all_words = []
        unique_tags = []
        training_data = []

        for intent in intents_data['intents']:
            current_tag = intent['tag']
            unique_tags.append(current_tag)

            for pattern in intent['patterns']:
                tokenized_words = self.tokenize_sentence(pattern)
                all_words.extend(tokenized_words)
                training_data.append((tokenized_words, current_tag))

        punctuation_to_ignore = ["?", "!", ".", ","]
        filtered_words = [self.stem(word) for word in all_words if word not in punctuation_to_ignore]
        unique_words = sorted(set(filtered_words))
        unique_tags = sorted(set(unique_tags))
