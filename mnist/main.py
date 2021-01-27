##############################
# main.py
##############################
# Description:
# * Read in local MNIST data,
# train neural network using data
# and labels, generate predictions
# and display accuracy.

from mnist import MNIST
from NN.neural_network import NeuralNetwork
import numpy as np
import os
from torch.utils.data import DataLoader

def load_data():
    """
    * Load data from local mnist dataset (in Data folder).
    """
    data = MNIST('Data')
    # images, labels:
    training = data.load_training()
    testing = data.load_testing()

    return training, testing

def transform_and_batch(training, testing):
    """
    * Transform the training and testing sets
    for usage in neural network.
    """
    training = DataLoader(training, batch_size = 60, shuffle = True)
    return training, testing

def train_model(training, input_len):
    """
    * Train model using training set.
    """
    model = NeuralNetwork(input_len)
    model.train(training, verbose = True)
    return model

def test_model(model, testing, report_path):
    """
    * Test model using testing set.
    """
    predictions = []
    for idx, input in enumerate(testing[0]):
        label = testing[1][idx]
        result = model.forward(input)
        # Convert the forward pass into a result:
        predictions.append(result)



def main():
    """
    * Perform key steps in order.
    """
    training, testing = load_data()
    training, testing = transform_and_batch(training, testing)
    model = train_model(training, len(testing[0][0]))
    test_model(model, testing, "MNIST_Predictions.csv")

if __name__ == '__main__':
    main()