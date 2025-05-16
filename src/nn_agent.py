import numpy as np

class NNAgent:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights = {
            'w1': np.random.randn(hidden_size, input_size),
            'w2': np.random.randn(output_size, hidden_size)
        }

    def forward(self, inputs):
        x = np.tanh(np.dot(self.weights['w1'], inputs))
        out = np.tanh(np.dot(self.weights['w2'], x))  # Output: throttle, steering
        return out

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights
