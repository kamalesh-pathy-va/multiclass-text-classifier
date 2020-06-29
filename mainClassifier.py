import sklearn
import json
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import random

with open("trainingData.json", "r") as file:
    data = json.load(file)

allWords = []
allSentences = []
allTags = []

for intend in data["intends"]:
    for sentence in intend["patterns"]:
        allSentences.append(sentence)
        allTags.append(intend['tag'])
