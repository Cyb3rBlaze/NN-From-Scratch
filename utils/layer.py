import numpy as np

from utils.number import Number


class Layer:
    def __init__(self, num_neurons, activation="linear"):
        self.num_neurons = num_neurons

        self.activation = activation

        self.weights = np.array([Number(i - 0.5) for i in np.random.rand(num_neurons)])
        self.biases = np.array([Number(i - 0.5) for i in np.random.rand(num_neurons)])

    def print_params(self):
        print("WEIGHTS:")
        print(self.weights)
        print("BIASES:")
        print(self.biases)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def d_sigmoid(x):
        return Layer.sigmoid(x) * (1 - Layer.sigmoid(x))

    @staticmethod
    def d_tanh(x):
        return 1 - np.square(Layer.tanh(x))

    def forward(self, input):
        if not isinstance(input, np.ndarray):
            raise TypeError("Input must be a NumPy array.")

        lin = np.dot(input, self.weights) + self.biases
