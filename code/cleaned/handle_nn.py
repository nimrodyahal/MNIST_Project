# -*- coding: utf-8 -*-
import cPickle
from neural_network import Network, ConvPoolLayer, FullyConnectedLayer, \
    SoftmaxLayer, relu, load_data_shared, MultiNet


def get_file_name():
    save_dir = '..\\Saved Nets'
    i = 0
    while True:
        file_name = 'test_net{}.txt'.format(i)
        yield save_dir + '\\' + file_name
        i += 1


def train_net(training_data, test_data, mapping):
    mini_batch_size = 10
    # net = Network([
    #     ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
    #                   filter_shape=(20, 1, 5, 5),
    #                   poolsize=(2, 2),
    #                   activation_fn=relu),
    #     ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
    #                   filter_shape=(40, 20, 5, 5),
    #                   poolsize=(2, 2),
    #                   activation_fn=relu),
    #     FullyConnectedLayer(
    #         n_in=40 * 4 * 4, n_out=1000, activation_fn=relu, p_dropout=0.5),
    #     FullyConnectedLayer(
    #         n_in=1000, n_out=300, activation_fn=relu, p_dropout=0.5),
    #     SoftmaxLayer(n_in=300, n_out=47, p_dropout=0.5)],
    #     mini_batch_size, mapping)
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
            n_in=1000, n_out=300, activation_fn=relu, p_dropout=0.5),
        SoftmaxLayer(n_in=300, n_out=47, p_dropout=0.5)],
        mini_batch_size, mapping)
    net.sgd(training_data, 30, mini_batch_size, 0.03, test_data)
    return net


def train_multi_net(net_count):
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
    nets = []
    for path in nets_path:
        nets.append(load_net(path))
    return MultiNet(nets)


def load_net(filename):
    with open(filename, 'rb') as f:
        return cPickle.load(f)