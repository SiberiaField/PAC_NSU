import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()
lr = 0.1
moment = 0

class Neuron:
    def __init__(self, num_of_weights):
        self.weights = rng.uniform(size=(1, num_of_weights))
        self.bias = rng.uniform()
        self.output = np.ndarray
        self.prev_diff = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x : np.ndarray) -> np.ndarray:
        y = np.dot(x, self.weights.T) + self.bias
        self.output = self.sigmoid(y)
        return self.output

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def backward(self, inputs : np.ndarray, loss : np.ndarray) -> np.ndarray:
        delta = self.sigmoid_derivative(self.output) * loss
        grad = np.dot(inputs.T, delta)
        diff = grad * lr + self.prev_diff * moment
        i = 0
        for w in self.weights:
            w += diff[0][i]
            i += 1
        self.prev_diff = diff
        self.bias += np.sum(delta)
        return np.dot(delta, self.weights)

class Model:
    def __init__(self):
        self.hidden_neurons = [Neuron(2) for _ in range(2)]
        self.output_neuron = Neuron(2)
        self.h_output = np.ndarray
    
    def forward(self, inputs : np.ndarray) -> np.ndarray:
        output_1 = self.hidden_neurons[0].forward(inputs)
        output_2 = self.hidden_neurons[1].forward(inputs)
        self.h_output = np.concat([output_1, output_2], axis=1)
        self.output_neuron.forward(self.h_output)
        return self.output_neuron.output
    
    def backward(self, output_loss : np.ndarray, inputs : np.ndarray): 
        hidden_loss = self.output_neuron.backward(self.h_output, output_loss)
        hidden_loss = hidden_loss.T
        self.hidden_neurons[0].backward(inputs, hidden_loss[0].reshape((4, 1)))
        self.hidden_neurons[1].backward(inputs, hidden_loss[1].reshape((4, 1)))

def train(m : Model, epochs : int, inputs : np.ndarray):
    expected_output = np.array([[0],[1],[1],[0]])
    for _ in range(epochs):
        predict = m.forward(inputs)
        output_loss = expected_output - predict
        m.backward(output_loss, inputs)

def print_weights(m : Model):
    print("h0", m.hidden_neurons[0].weights)
    print("h1", m.hidden_neurons[1].weights)
    print("o", m.output_neuron.weights)
    print()

def main():
    inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    m = Model()

    print("\nWeights before train:")
    print_weights(m)

    train(m, 10000, inputs)

    print("Weights after train:")
    print_weights(m)

    print(f"Result:\n{m.forward(inputs)}\n")
    plt.show()
main()