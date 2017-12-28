# -*- coding: utf-8 -*-
from neu_net_test import Network, load_net, FullyConnectedLayer, \
    SoftmaxLayer, relu, load_data_shared
from PIL import Image
import numpy as np


def test_by_hand():
    img_to_bytes()
    with open('test_byte_stuff', 'r') as f:
        bits = map(int, list(str(f.read())))
    # a = np.zeros((784, 1))
    a = np.zeros((1, 784))
    for index, val in zip(a, bits):
        index[0] = val
    return a


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


def main():
    training_data, validation_data, test_data = load_data_shared()
    mini_batch_size = 10
    # expanded_training_data, _, _ = load_data_shared(
    #     "../data/mnist_expanded.pkl.gz")
    net = Network([
        FullyConnectedLayer(
            n_in=28 * 28, n_out=16, activation_fn=relu, p_dropout=0.5),
        SoftmaxLayer(n_in=16, n_out=10, p_dropout=0.5)],
        mini_batch_size)
    net.sgd(training_data, 1, mini_batch_size, 0.03,
            validation_data, test_data)
    net.save('Saved\\test_net.txt')
    print 'finished learning'
    del net
    net = load_net('Saved\\test_net.txt')
    print 'predictions:  ', net.test_mb_predictions(0)
    print 'real:         ', net.print_real_values(0)
    print 'bla:', net.feedforward(test_by_hand())


if __name__ == '__main__':
    main()