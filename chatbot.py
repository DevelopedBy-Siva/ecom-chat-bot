import sys
import json
import nltk
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def train_model():
    if len(sys.argv) > 1:
        print(sys.argv[1])

class Bot:
    def __init__(self):
        train_model()

