import numpy as np
import random


## Sigmoid function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


## Derivative of sigmoid function
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    """  Neural network class

    Attributes:
        sizes: List specifying number of neurons in each layer of the neural network 
                (Ex. [2 4 3] - network with 2 neurons in first (input) layer, 4 neurons in second layer, and 3 neurons in third layer)
        num_layers: Integer, number of layers in the network
        biases: List; ith element contains column-vector with biases for i+2 layer of neural network; 
                biases are omitted for input layer
        weights: List; ith element contains matrix of weights connecting i+1 and i+2 layers of the neural network              


    """

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """ Calculates output of the neural network given input a

        :param a: numpy [n,1] vector, input to the neural network     
        :return: output of the neural network
        """

        for (b, w) in zip(self.biases, self.weights):
            a = sigmoid(w.dot(a) + b)

        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """ Training neural network using stochastic gradient descent

        :param training_data: list of tuples (x, y), where x is training input and y is a label
        :param epochs: integer, number of SGD epochs
        :param mini_batch_size: integer, size of mini batch
        :param eta: float, learning rate of gradient descent mechanism
        :param test_data: list of tuples (x, y), where x is test input and y is a label
        :return: Updates weight and biases in the neural network
        """

        if test_data: n_test = len(test_data)
        n = len(training_data)

        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def cost_derivative(self, output_activations, y):
        """

        :param output_activations: 
        :param y: 
        :return: vector of output derivatives partial C/partial a_j
        """
        return output_activations - y

    def update_mini_batch(self, mini_batch, eta):
        """
        Updating weights and biases in the neural network by applying backpropagation mechanism to a single mini batch
        :param mini_batch: list of tuples containing mini batch of a training data
        :param eta: learning rate
        :return: updates weights and biases
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        ## transform mini_batch list into numpy array, with each training example in a column
        mini_batch_array = np.array(mini_batch)
        x_matr = np.concatenate(mini_batch_array[:,0], axis=1)
        y_matr = np.concatenate(mini_batch_array[:,1], axis=1)

        ## Feedforward
        activation = x_matr
        activations = [x_matr]
        zs = []

        for w, b in zip(self.weights, self.biases):
            z = w.dot(activation) + b
            activation = sigmoid(z)
            activations.append(activation)
            zs.append(z)

        ## Backpropagation
        delta = self.cost_derivative(activations[-1], y_matr) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta.sum(axis=1)[:, None]
        nabla_w[-1] = np.einsum('ik,jk->ji', activations[-2], delta)

        for l in xrange(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(zs[-l])
            nabla_b[-l] = delta.sum(axis=1)[:, None]
            nabla_w[-l] = np.einsum('ik,jk->ji', activations[-l-1], delta)

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)