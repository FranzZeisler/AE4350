import numpy as np

class NeuralController:
    """
    A simple feedforward neural network controller for the car.
    It takes a genome (flattened weights and biases) and uses it to
    compute the steering and throttle outputs based on the input features.
    """
    def __init__(self, genome, input_dim=18, hidden_dim=32, output_dim=2):
        """
        Initialize the neural network with genome weights and biases.
        :param genome: Flattened weights and biases of the neural network.
        :param input_dim: Number of input features.
        :param hidden_dim: Number of neurons in the hidden layer.
        :param output_dim: Number of output features (steering and throttle).
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights and biases from the genome
        self.weights1 = genome[0:input_dim*hidden_dim].reshape((input_dim, hidden_dim))
        self.bias1 = genome[input_dim*hidden_dim:input_dim*hidden_dim + hidden_dim]

        # The second layer weights and biases
        # The output layer has 2 outputs (steering and throttle)
        offset = input_dim * hidden_dim + hidden_dim
        self.weights2 = genome[offset:offset + hidden_dim*output_dim].reshape((hidden_dim, output_dim))
        self.bias2 = genome[offset + hidden_dim*output_dim:]

    def relu(self, x):
        """
        ReLU activation function.
        """
        return np.maximum(0, x)

    def forward(self, inputs):
        """
        Forward pass through the neural network.
        :param inputs: Input features (flattened).
        :return: Steering and throttle outputs.
        """
        x = self.relu(np.dot(inputs, self.weights1) + self.bias1) # ReLU → [0, ∞)
        x = np.tanh(np.dot(x, self.weights2) + self.bias2)  # tanh → [-1, 1]
        return x  # [steering, throttle]
