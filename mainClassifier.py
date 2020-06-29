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


# Splitting the data for training and testng
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(allSentences, allTags, test_size=0.1)


# Creating the classifier.
model = Pipeline([('vect', CountVectorizer()),
                  ('tfidf', TfidfTransformer()),
                  ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                  ])

# Training the model.
model.fit(x_train, y_train)  # Use this line if you are training and testing the model
# model.fit(allSentences, allTags)  # Use this line if you are training with all the data and actually trying to implement the classifier in real world.

# Saving the model.
with open("thechatbot.pickle", "wb") as f:
    pickle.dump(model, f)

# Loading the pre-trained model. If you dont want to train the model every time the code runs PLEASE comment out the "model.fit" lines and also the "Model saving lines"
with open("thechatbot.pickle", "rb") as theModelsFile:
    model = pickle.load(theModelsFile)


def classifyText(inputText):
	"""
	I'll add comments later
	"""
	theIN = [inputText]
	prediction = model.predict(theIN)
	predictionConfidence = model.predict_proba(theIN)
	maxConfidence = max(predictionConfidence[0])
	if maxConfidence > 0.8:  # This if condition is to return the prediction if the accuracy is more than 80%
	    return prediction[0]
	else:
	    return False


def main():
	"""
	I'll add comments later
	"""
	prediction = model.predict(x_test)
	predictionConfidence = model.predict_proba(x_test)
	for i in range(0,len(x_test)):
		print(f'Actual Text: {x_test[i]}, Predicted catagory: {prediction[i]}, Actual catagory: {y_test[i]}, Accuracy: {max(predictionConfidence[i])}')
		print()


if __name__ == '__main__':
	print("Please import this script into your project and use the ClassifyText() method if you aren't testing")
	main()