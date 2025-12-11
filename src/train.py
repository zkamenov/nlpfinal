import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys

from nltk.tokenize import WordPunctTokenizer

class SequenceNeuralNetwork(nn.Module):
	def __init__(self, vocab, prefixSize, vectorSize=10, hiddenDim=10):
		super(SequenceNeuralNetwork, self).__init__()

		self.indices = {} #Keeps track of words as indices since pytorch only accepts numbers, not strings
		self.words = {}

		i = 0
		for key in vocab: #Builds indices from vocabulary
			self.indices[key] = i
			self.words[i] = key
			i += 1

		self.prefixSize = prefixSize

		self.emb = nn.Embedding(len(vocab), vectorSize) #Layers
		self.lin1 = nn.Linear(vectorSize, hiddenDim)
		self.relu = nn.ReLU()
		self.lin2 = nn.Linear(hiddenDim, len(vocab))
		self.sigmoid = nn.Sigmoid()

	def forward(self, x): #Forward pass
		emb = self.emb(x)
		lin1 = self.lin1(emb)
		relu = self.relu(lin1)
		lin2 = self.lin2(relu)
		sigmoid = self.sigmoid(lin2)
		return sigmoid

	def prefixSequence(self, sequence): #Generates n-grams
		x = []
		y = []

		for i in range(0, len(sequence)-self.prefixSize):
			prefix = []
			for j in range(i,i+self.prefixSize):
				prefix.append(self.indices[sequence[j]])
			x.append(prefix)

			targetProbability = [0]*len(self.indices) #Target vector is probability of each word (1 for the target and 0 for everything except the target)
			targetProbability[self.indices[sequence[i+self.prefixSize]]] = 1
			y.append(targetProbability)
		return (torch.tensor(x), torch.tensor(y));

	def trainModel(self, trainSequence, num_epochs=100, learning_rate=0.001): #Trains the model
		(x_train, y_train) = self.prefixSequence(trainSequence) #Generates training data

		loss_fn = nn.CrossEntropyLoss()
		optimizer = optim.Adam(self.parameters(), lr=learning_rate)

		for i in range(num_epochs):
			optimizer.zero_grad()
			predicted_probabilities = self.forward(x_train)
			loss = loss_fn(predicted_probabilities, y_train)
			print(f"Training epoch number {i+1}/{num_epochs}, loss function:{loss}")


			loss.backward()
			optimizer.step()

	def predictWordIndex(self, prefixIndices): #Predicts the word index based on the input
		outputProbabilities = torch.argmax(self(prefixIndices), dim=1)
		record = 0.0
		recordIndex = 0
		for i in range(len(outputProbabilities)):
			if (outputProbabilities[i] > record):
				record = outputProbabilities[i] #Locates word with highest probability
				recordIndex = i
		return (recordIndex, record)

	def predictWord(self, prefix): #Wrapper function for Predicting the Word (Converts Word to index and back again for the output)
		prefixIndices = []
		for word in prefix:
			prefixIndices.append(self.indices[word])
		predictedIndex = self.predictWordIndex(torch.tensor(prefixIndices))
		predictedWord = (self.words[predictedIndex[0]], predictedIndex[1])
		return predictedWord

	def evaluate(self, testSequence): #Performs evaluation metrics
		self.eval()
		accuracy = 0.0

		(x_test, y_test) = self.prefixSequence(testSequence) #Generates testing n-grams

		with torch.no_grad():
			score = 0
			for i in range(len(x_test)): #Calculates "Accuracy"
				if (self.predictWordIndex(x_test[i])[0] == self.predictWordIndex(y_test[i])[0]):
					score += 1

			accuracy = score / len(x_test)

		return accuracy




if __name__ == "__main__":
	lines = []

	infile = '../data/mexicous.txt' #File path of the data file

	with open(infile, 'r', encoding="utf-8") as f:
		lines = f.readlines() #Reads input file (first line is training data and second line is testing data)

	tk = WordPunctTokenizer()
	trainSequence = tk.tokenize(lines[0]) #Tokenizes the training data
	testSequence = tk.tokenize(lines[1]) #Tokenizes the testing data
	generalSequence = trainSequence.copy()
	generalSequence.extend(testSequence)

	vocab = {}
	for token in generalSequence:
		if token in vocab: #Generates vocabulary
			vocab[token] += 1
		else:
			vocab[token] = 1

	prefixLength = 3 #Prefix length

	network = SequenceNeuralNetwork(vocab, prefixLength)

	network.trainModel(trainSequence)
	accuracy = network.evaluate(testSequence)
	print(f"Accuracy: {accuracy}\n") #Displays "Accuracy" of the data

	print("Enter prefix: ")
	for line in sys.stdin:
		word = network.predictWord(tk.tokenize(line)) #Predicts word based on the standard input
		print(f"PredictedWord: {word}")
