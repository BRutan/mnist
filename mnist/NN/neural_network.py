##############################
# neural_network.py
##############################
# Description:
# * Neutral network for predicting
# digits in MNIST dataset.

from tqdm import trange
from torch.optim import SGD
from torch.nn import Linear, Module, MSELoss, ReLU, Sequential, Sigmoid

class NeuralNetwork(Module):
    """
    * Neural network for training
    and testing MNIST dataset.
    """
    def __init__(self, dims, loss = MSELoss):
        """
        * Initialize model corresponding
        to input dimensions and that uses
        a particular learning rate.
        """
        self.__model = Sequential([Linear(), ReLU, Linear(), ReLU])
        self.__loss = loss

    ###############
    # Interface Methods:
    ###############
    def train(self, batches, epochs = 50, lr = .0001, verbose = False):
        """
        * Train model using training batches.
        Inputs:
        * batches: Batcher with appropriately transformed
        datasets for neural network model.
        """
        self.__optimizer = SGD(self.__model.parameters(), lr = lr)
        epoch_range = trange(0, epochs) if verbose else range(0, epochs)
        for epoch in epoch_range:
            for input, target in batches:
                self.__optimizer.zero_grad()
                output = self.__model(input)
                loss = self.__loss(output, target)
                loss.backward()
                self.__optimizer.step()

    def forward(self, data):
        """
        * Pass data through the neural network
        to generate an (untransformed) prediction.
        """
        pass