import random

import numpy as np


class Network:

    def __init__(self, sizes):
        self.layers = sizes
        self.num_of_layers = len(self.layers)
        self.biases = [np.zeros((layer_size, 1)) for layer_size in self.layers[1:]]
        self.weights = [np.random.rand((layer_size, prev_layer_size)) for layer_size, prev_layer_size in
                        zip(self.layers[1:], self.layers[:-1])]

    def feedforward(self, input):
        output = np.asarray([[x] for x in input])
        for layer in range(self.num_of_layers - 1):
            output = self.sigmoid(np.dot(self.weights[layer], output) + self.biases[layer])
        return output, output.argmax()

    def train(self, learning_rate, training_data, batch_size):
        random.shuffle(training_data)
        batch = [data for data in training_data[:batch_size]]
        for (input, label) in batch:
            # feed forward
            outputs = []
            activations = [input]
            output = input
            for layer in range(self.num_of_layers - 1):
                output = np.dot(self.weights[layer], output) + self.biases[layer]
                outputs.append(outputs)
                activation = self.sigmoid(output)
                activations.append(activation)
            # backpropagation
            delta_biases = [np.zeros(bias.shape) for bias in self.biases]
            delta_weights = [np.zeros(weight.shape) for weight in self.weights]
            cost_derivative = activations[-1] - label
            activation = self.sigmoid(outputs[-1])
            activation_derivative = activation * (1 - activation)
            weight_derivative = activations[-2]
            delta = cost_derivative * activation_derivative
            delta_bias = delta
            delta_weight = np.dot(delta, weight_derivative.transpose())
            delta_biases[-1] = delta_bias
            delta_weights[-1] = delta_weight
            for layer in range(2, self.num_of_layers):
                activation = self.sigmoid(outputs[-layer])
                activation_derivative = np.dot(activation, (1 - activation))
                delta = np.dot(self.weights[-layer + 1].transpose(), delta) * activation_derivative
                delta_bias = delta
                delta_weight = np.dot(delta, activations[-layer - 1].transpose())
                delta_biases[-layer] = delta_biases[-layer] + delta_bias
                delta_weights[-layer] = delta_weights[-layer] + delta_weight
            # TODO refactor so that delta_bias is not shadowed
            self.biases = [bias - (learning_rate * delta_bias) / batch_size for bias, delta_bias in
                           zip(self.biases, delta_biases)]
            self.weights = [weight - (learning_rate * delta_weight) / batch_size for weight, delta_weigh in
                            zip(self.weights, delta_weights)]

    def sigmoid(self, layer):
        return 1 / (1 + np.exp(-layer))
