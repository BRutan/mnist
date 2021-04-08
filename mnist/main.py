##############################
# main.py
##############################
# Description:
# * Read in local MNIST data,
# train neural network using data
# and labels, generate predictions
# and display accuracy.

import matplotlib.pyplot as plt
from NN.neural_network import MNISTPredictor
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from torch import flatten, FloatTensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

normalize_data = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), flatten])
#target_transform = transforms.Compose([])

def load_data(batchsize):
    """
    * Load data from local mnist dataset (in Data folder).
    """
    train = DataLoader(datasets.MNIST('data', train = True, download = True, transform = normalize_data), batch_size = batchsize)
    test = DataLoader(datasets.MNIST('data', train = False, transform = normalize_data))
    return train, test

def train_model(train, input_nodes, hidden_nodes, num_outputs, epochs, lr, track):
    """
    * Train model using training set.
    """
    model = MNISTPredictor(input_nodes, hidden_nodes, num_outputs)
    losses = model.train(train, 1, lr, verbose = True, track = track)
    return model, losses

def plot_losses(losses, epochs, batchsize, lr):
    """
    * Plot losses by batch for visualization purposes.
    """
    plt.clf()
    plt.title('MNIST Dataset Losses Epochs:%s BatchSize:%s LR:%s' % (epochs, batchsize, lr))
    plt.scatter(list(range(len(losses))), losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig('Losses_%s_%s_%s.png' % (epochs, batchsize, str(lr).strip('.')))

def test_model(model, test, report_path):
    """
    * Test model using testing set.
    """
    labels = []
    predicted = []
    with open(report_path, 'w') as f:
        f.write('Actual,Predicted')
        for input, target in test:
            pred = model.forward(input)
            pred = model.convert_pass(pred)
            # Convert the forward pass into a result:
            labels.append(target)
            predicted.append(predicted)
            f.write('%s,%s' % (target, pred))
    return confusion_matrix(predicted, labels)

def main():
    """
    * Perform key steps in order.
    """
    epochs = 1
    batchsize = 60
    lr = .01
    train, test = load_data(batchsize)
    model, losses = train_model(train, 784, (128, 64), 10, epochs, lr, True)
    plot_losses(losses, epochs, batchsize, lr)
    matr = test_model(model, test, "MNIST_Predictions.csv")
    print(matr)

if __name__ == '__main__':
    main()