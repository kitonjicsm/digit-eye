import numpy as np


class Network:

    def __init__(self, sizes):
        self.layers = sizes
        self.num_of_layers = len(sizes)
        self.biases = [np.zeros((layer_size, 1)) for layer_size in sizes[1:]]
        self.weights = [np.zeros((layer_size, prev_layer_size)) for layer_size, prev_layer_size in
                        zip(sizes[1:], sizes[:-1])]

    def feedforward(self, input):
        output = np.asarray([[x] for x in input])
        for layer in range(self.num_of_layers - 1):
            output = self.sigmoid(np.dot(self.weights[layer], output) + self.biases[layer])
        return output, output.argmax()

    def sigmoid(self, layer):
        return 1 / (1 + np.exp(-layer))
