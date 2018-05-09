# -*- coding: utf-8 -*-
from neural_network import *


def get_file_name():
    """
    Generator the returns free file names.
    """
    save_dir = '..\\Saved Nets'
    i = 0
    while True:
        file_name = 'test_net{}.txt'.format(i)
        yield save_dir + '\\' + file_name
        i += 1


def train_net(training_data, test_data, mapping):
    """
    Trains a deep convolutional neural network with 2 convolutional layers, 2
    pooling layers, 2 fully connected layers and a softmax layer. Uses dropout
    for normalization, a ReLu activation function, a minibatch size of 10, a
    learning rate of 0.3, 40 epochs, and Stochastic Gradient Descent method.
    :param training_data: The training dataset.
    :param test_data: The testing dataset.
    :param mapping: The mapping.
    :return: A trained neural network.
    """
    mini_batch_size = 10
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=relu),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=relu),
        FullyConnectedLayer(
            n_in=40 * 4 * 4, n_out=1000, activation_fn=relu, p_dropout=0.5),
        FullyConnectedLayer(
            n_in=1000, n_out=1000, activation_fn=relu, p_dropout=0.5),
        SoftmaxLayer(n_in=1000, n_out=47, p_dropout=0.5)],
        mini_batch_size, mapping)
    net.sgd(training_data, 40, 0.03, test_data)
    return net


def train_multi_net(net_count):
    """
    Trains a multi-network.
    :param net_count: The amount of neural networks in the multi-net.
    :returns: The multi-net.
    """
    training_data, test_data, mapping = load_data_shared()
    file_name = get_file_name()
    paths = []
    for i in range(net_count):
        print 'Training net{}:'.format(i)
        net = train_net(training_data, test_data, mapping)
        name = file_name.next()
        paths.append(name)
        net.save(name)
        print 'Finished Training net' + str(i)
    return load_multi_net(paths)


def load_multi_net(nets_path):
    """
    Loads a multi-net.
    :param nets_path: [path of neural network]
    :returns: The multi-net.
    """
    nets = []
    for path in nets_path:
        nets.append(load_net(path))
    return MultiNet(nets)


def load_net(filename):
    """
    Loads a single neural network.
    :param filename: The of the neural network.
    :returns: The neural network.
    """
    with open(filename, 'rb') as f:
        return cPickle.load(f)