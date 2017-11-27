# -*- coding: utf-8 -*-
import mnist_loader
from neural_network import Network


def main():
    training_data, validation_data, test_data = \
        mnist_loader.load_data_wrapper()
    sizes = [784, 30, 10]
    net = Network(sizes)
    learning_rate = 0.25
    lmbda = 5
    net.stochastic_gradient_descent(training_data, 10, learning_rate,
                                    lmbda, test_data=test_data)

    # training_data, validation_data, test_data = \
    #     mnist_loader.load_data_wrapper()
    # sizes = [784, 10]
    # net = Network(sizes)
    # learning_rate = 1.0
    # lmbda = 20
    # net.stochastic_gradient_descent(training_data[:1000], 30, 10,
    #                                 learning_rate, lmbda,
    #                                 test_data=test_data[:100])


if __name__ == '__main__':
    main()