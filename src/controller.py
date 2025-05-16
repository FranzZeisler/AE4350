import numpy as np

class SimpleNNController:
    def __init__(self, weights):
        self.w = weights.reshape((2, 2))  # 2 inputs → 2 outputs

    def act(self, inputs):
        return np.tanh(np.dot(inputs, self.w))  # output = [steering, throttle]
    
    