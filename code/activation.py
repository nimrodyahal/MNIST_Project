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


def test_by_hand(net):
    img_to_bytes()
    with open('test_byte_stuff', 'r') as f:
        bits = map(int, list(str(f.read())))
    a = np.zeros((784, 1))
    for index, val in zip(a, bits):
        index[0] = val
    print np.argmax(net.feedforward(a))


def main():
    net = neural_network.load(get_file_name())
    test_by_hand(net)
    # activate()
    # training_data, validation_data, test_data = \
    #     mnist_loader.load_data_wrapper()
    # sizes = [784, 100, 100, 10]
    # time1 = time.time()
    # print test_data[0][0]
    # print test_data[0][1]
    # time2 = time.time()
    # print time2 - time1 * 1000.0


if __name__ == '__main__':
    main()