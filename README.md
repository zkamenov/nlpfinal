# How to operate the program

Ensure you have the proper libraries installed.

Run:

$ python3 train.py

Ater running the program, the model trains.

After training is complete, accuracy is printed to the standard output.

After this, you can type a prefix of words, however the prefix must be composed of words that are in the vocabulary of the data, and must be at least the length of the prefix length constant (default value is 3).

#Variables

To change the behavior of the program, some variables will need to be changed.

The inFile variable shows the data file path. Change it to represent any other file in the data folder

Prefix length, number of epochs, and learning rate are also constants defined the code.

#Testing other dataset

Dataset must be converted into a text file with only 2 lines.

The first line will be the training data and the second line will be the testing data.

Input into the data folder & set the file path in the train.py
