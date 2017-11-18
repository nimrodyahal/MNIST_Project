# -*- coding: utf-8 -*-
import numpy as np
import random


def sigmoid(z):
    """
    Applies sigmoid function to numpy array or number.
    :param z: Numpy array or number.
    :return: The array after applying the sigmoid function.
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Applies the derivative of the sigmoid to numpy array or number.
    :param z: Numpy array or number.
    :return: The derivative of the sigmoid function.
    """
    return sigmoid(z) * (1 - sigmoid(z))


class Network():
    def __init__(self, sizes):
        """
        :param sizes: A list containing the number of neurons in each layer
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # A numpy
        # array of the bias of each neuron
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]  # A numpy
        # array of the weight of each synapse

    def feedforward(self, prev_layer):
        """
        Feeds forward to the next layer.
        :param prev_layer: The previous layer
        :return: The next layer
        """
        current_layer = np.array([])
        for b, w in zip(self.biases, self.weights):
            current_layer = sigmoid(np.dot(w, prev_layer) + b)
        return current_layer

    def stochastic_gradient_descent(self, training_data, epochs,
                                    mini_batch_size, learning_rate,
                                    test_data=None):
        """
        Train the neural network using mini-batch stochastic
        gradient descent.
        :param training_data: A list of tuples "(x, y)" representing the
        training inputs and the desired outputs.
        :param epochs: The number of times the network trains.
        :param mini_batch_size: The size of the mini-batches.
        :param learning_rate: The learning rate.
        :param test_data: If provided then the network will be evaluated
        against the test data after each epoch, and partial progress printed
        out. useful for tracking progress, but slows things down substantially.
        :return: None.
        """
        for epoch in xrange(epochs):
            random.shuffle(training_data)  # Shuffles the training data
            mini_batches = [training_data[k:k + mini_batch_size] for k in
                            xrange(0, len(training_data), mini_batch_size)]
            # Separate the training data to mini-batches

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)  # Train
                # network

            if test_data:
                print "Epoch {0}: {1} / {2}".format(epoch,
                                                    self.evaluate(test_data),
                                                    len(test_data))
            else:
                print "Epoch {0} complete".format(epoch)

    def update_mini_batch(self, mini_batch, learning_rate):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        :param mini_batch: A list of tuples "(x, y)" representing the training
        inputs and the desired outputs.
        :param learning_rate: The learning rate.
        :return: None
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)  # Compute cost
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]  # Update
        # weights
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]  # Update biases

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    @staticmethod
    def cost_derivative(output_activations, y):
        """
        Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations.
        """
        return output_activations - y
