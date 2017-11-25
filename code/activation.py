# -*- coding: utf-8 -*-
import mnist_loader
from neural_network import Network, load


def main():
    training_data, validation_data, test_data = \
        mnist_loader.load_data_wrapper()
    sizes = [784, 30, 10]
    net = Network(sizes)
    learning_rate = 10.0
    lmbda = 1000
    net.stochastic_gradient_descent(training_data, 30, 10, learning_rate,
                                    lmbda, test_data=test_data)


if __name__ == '__main__':
    main()