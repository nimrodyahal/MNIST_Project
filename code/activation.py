# -*- coding: utf-8 -*-
import mnist_loader
from neural_network import Network
import neural_network
import time
import numpy as np


def get_file_name():
    save_dir = 'Saved'
    file_name = 'test_net.txt'
    return save_dir + '\\' + file_name


def activate():
    training_data, validation_data, test_data = \
        mnist_loader.load_data_wrapper()
    sizes = [784, 100, 100, 10]
    net = Network(sizes)
    learning_rate = 0.25
    lmbda = 5
    net.stochastic_gradient_descent(training_data, 10, learning_rate,
                                    lmbda, test_data=test_data)
    net.save_best(get_file_name())


def main():
    # activate()
    training_data, validation_data, test_data = \
        mnist_loader.load_data_wrapper()
    # sizes = [784, 100, 100, 10]
    net = neural_network.load(get_file_name())
    time1 = time.time()
    print np.argmax(net.feedforward(test_data[0][0]))
    print test_data[0][1]
    time2 = time.time()
    print time2 - time1 * 1000.0


if __name__ == '__main__':
    main()