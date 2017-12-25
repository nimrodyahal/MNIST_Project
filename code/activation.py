# -*- coding: utf-8 -*-
import mnist_loader
from neural_network import Network
import neural_network
import time
import numpy as np
from PIL import Image


def img_to_bytes():
    with open('test_byte_stuff', 'w') as f:
        im = Image.open("test_draw.png")
        pix = im.load()
        for y in xrange(28):
            for x in xrange(28):
                if pix[x, y] == (0, 0, 0):
                    f.write('1')
                else:
                    f.write('0')


def get_file_name():
    save_dir = 'Saved'
    i = 0
    while True:
        file_name = 'test_net{}.txt'.format(i)
        yield save_dir + '\\' + file_name
        i += 1


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


def test_by_hand(net):
    img_to_bytes()
    with open('test_byte_stuff', 'r') as f:
        bits = map(int, list(str(f.read())))
    a = np.zeros((784, 1))
    for index, val in zip(a, bits):
        index[0] = val
    b = net.feedforward(a)
    print b
    print np.argmax(b)


def testeste(filename):
    from neu_net_test import Network, ConvPoolLayer, FullyConnectedLayer, \
        SoftmaxLayer, relu, load_data_shared
    training_data, validation_data, test_data = load_data_shared()
    mini_batch_size = 10
    expanded_training_data, _, _ = load_data_shared(
        "../data/mnist_expanded.pkl.gz")
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
        SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)],
        mini_batch_size)
    net.sgd(expanded_training_data, 40, mini_batch_size, 0.03,
            validation_data, test_data)
    net.save(filename)


def main():
    file_names = get_file_name()
    testeste(file_names.next())
    testeste(file_names.next())
    testeste(file_names.next())
    testeste(file_names.next())
    testeste(file_names.next())
    # net = neural_network.load(get_file_name())
    # test_by_hand(net)
    # activate()
    # training_data, validation_data, test_data = \
    #     mnist_loader.load_data_wrapper()
    # sizes = [784, 100, 100, 10]
    # time1 = time.time()
    # time2 = time.time()
    # print (time2 - time1) * 1000.0


if __name__ == '__main__':
    main()