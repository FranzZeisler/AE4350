import numpy as np

class NeuralController:
    def __init__(self, genome, input_dim=18, hidden_dim=32, output_dim=2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.weights1 = genome[0:input_dim*hidden_dim].reshape((input_dim, hidden_dim))
        self.bias1 = genome[input_dim*hidden_dim:input_dim*hidden_dim + hidden_dim]

        offset = input_dim * hidden_dim + hidden_dim
        self.weights2 = genome[offset:offset + hidden_dim*output_dim].reshape((hidden_dim, output_dim))
        self.bias2 = genome[offset + hidden_dim*output_dim:]

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, inputs):
        x = self.relu(np.dot(inputs, self.weights1) + self.bias1)
        x = np.tanh(np.dot(x, self.weights2) + self.bias2)  # tanh → [-1, 1]
        return x  # [steering, throttle]
