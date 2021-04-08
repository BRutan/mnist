##############################
# neural_network.py
##############################
# Description:
# * Neutral network for predicting
# digits in MNIST dataset.

from tqdm import trange
from torch import flatten, float64, logit, zeros
from torch.nn import Linear, Module, BCEWithLogitsLoss, ReLU, Sequential, Sigmoid, Softmax
from torch.optim import SGD

class MNISTPredictor(Module):
    """
    * Neural network for training
    and testing MNIST dataset.
    """
    def __init__(self, input_nodes, hidden_nodes, output_nodes, loss = BCEWithLogitsLoss()):
        """
        * Initialize model corresponding
        to input dimensions and that uses
        a particular learning rate.
        Inputs:
        * input_nodes: Number of input nodes (dimensions of each sampled tensor).
        * hidden_nodes: tuple containing (hidden_layer_node_1, ..., hidden_layer_node_n).
        * output_nodes: Number of output nodes (dimensions of output tensor).
        * loss: Loss function.
        """
        super().__init__()
        sequence = self.__GenerateSequence(input_nodes, hidden_nodes, output_nodes)
        self.__model = Sequential(*sequence)
        self.__loss = loss

    ###############
    # Properties:
    ###############
    @property
    def InputSize(self):
        return self.__inputsize
    @property
    def OutputSize(self):
        return self.__outputsize
    ###############
    # Interface Methods:
    ###############
    def encode_label(self, label):
        """
        * Transform numeric label to tensor for appropriate usage
        in gradient descent.
        """
        if label.shape[0] > 1:
            encoded = zeros([label.shape[0], 10], dtype=float64)
            for num, lbl in enumerate(label):
                encoded[num][lbl] = 1
        else:
            encoded = zeros([10], dtype=float64)
            encoded[label[0]] = 1
        return encoded

    def convert_pass(self, fwd):
        """
        * Convert output probabilities into
        prediction.
        """
        num = 0
        max_prob = fwd.max().item()
        for num in range(len(fwd)):
            if fwd[num][0].item() == max_prob:
                return num

    def train(self, batches, epochs = 50, lr = .01, verbose = False, track = False):
        """
        * Train model using training batches.
        Inputs:
        * batches: Batcher with appropriately transformed
        datasets for neural network model.
        """
        if track:
            losses = []
        self.__optimizer = SGD(self.__model.parameters(), lr = lr)
        epoch_range = trange(0, epochs) if verbose else range(0, epochs)
        batch_range = trange(0, len(batches)) if verbose else range(0, len(batches))
        for epoch in epoch_range:
            batch_iter = iter(batches)
            for batch in batch_range:
                input, target = next(batch_iter)
                target = self.encode_label(target)
                self.__optimizer.zero_grad()
                output = self.__model(input)
                loss = self.__loss(flatten(output), target)
                loss.backward()
                self.__optimizer.step()
                if track:
                    losses.append(loss.item())
        if track:
            return losses

    def forward(self, data):
        """
        * Pass data through the neural network
        to generate an (untransformed) prediction.
        """
        if data.shape[1] != self.__inputsize:
            raise Exception('data must have .shape[1] == %s' % self.__inputsize)
        return self.__model(data)

    def __GenerateSequence(self, input_nodes, hidden_nodes, output_nodes):
        """
        * Generate layers for neural network.
        """
        self.__inputsize = input_nodes
        self.__outputsize = output_nodes
        layers = []
        for num, hidden_dim in enumerate(hidden_nodes):
            if num == 0:
                layers.append(Linear(input_nodes, hidden_dim))
            else:
                layers.append(Linear(hidden_nodes[num - 1], hidden_dim))
        layers.append(Linear(hidden_dim, output_nodes))
        sequence = []
        for layer in layers:
            sequence.append(layer)
            sequence.append(Sigmoid())
        sequence.append(Softmax())
        return sequence