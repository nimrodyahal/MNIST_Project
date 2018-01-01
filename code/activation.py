# -*- coding: utf-8 -*-
import mnist_loader
from neural_network import Network
import neural_network
import time
import numpy as np
from PIL import Image
from scipy import ndimage
from neu_net_test import Network, ConvPoolLayer, FullyConnectedLayer, \
    SoftmaxLayer, relu, load_data_shared, MultiNet, load_net


def img_to_bytes():
    with open('test_byte_stuff', 'w') as f:
        im = Image.open("test_draw.png").convert('LA')
        pix = im.load()
        for y in xrange(28):
            for x in xrange(28):
                intensity = 1 - (pix[x, y][0] / 255.0)
                f.write(str(intensity) + ' ')
                # if pix[x, y] == (0, 255):
                #     f.write('1')
                # else:
                #     f.write('0')


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


def get_bounding_box(img):
    rows = len(img)
    cols = len(img[0])
    min_x = cols
    min_y = rows
    max_x = -1
    max_y = -1
    for y in range(rows):
        for x in range(cols):
            if img[y][x] > 0.01:
                if min_x > x:
                    min_x = x
                if max_x < x:
                    max_x = x
                if min_y > y:
                    min_y = y
                if max_y < y:
                    max_y = y
    return min_x, min_y, max_x, max_y


def test_by_hand():
    img_to_bytes()
    with open('test_byte_stuff', 'r') as f:
        bits = map(float, str(f.read()).split(' ')[:-1])
    canvas = np.zeros((784, 1))
    for index, val in zip(canvas, bits):
        index[0] = val
    canvas = canvas.reshape((28, 28))
    # Get bounding box of char
    min_x, min_y, max_x, max_y = get_bounding_box(canvas)
    boundbox_x = max_x - min_x + 1
    boundbox_y = max_y - min_y + 1
    scaling = 20. / max(boundbox_x, boundbox_y)
    zoomed_char = ndimage.interpolation.zoom(canvas[min_y:max_y, min_x:max_x],
                                             scaling)  # Rescale the char to
                                             #  be 20 * 20 pixels
    #
    canvas = np.zeros((28, 28))
    zm_chr_shape = zoomed_char.shape
    canvas[0:zm_chr_shape[0], 0:zm_chr_shape[1]] = zoomed_char
    center = ndimage.measurements.center_of_mass(canvas)
    shift = (28 / 2. - center[0], 28 / 2. - center[1])
    canvas = ndimage.interpolation.shift(canvas, shift).reshape((784, 1))
    print canvas
    return canvas
    # b = net.feedforward(a)
    # print b
    # print np.argmax(b)


def train_net(expanded_training_data):
    training_data, validation_data, test_data = load_data_shared()
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
        SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)],
        mini_batch_size)
    # net = Network([
    #     ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
    #                   filter_shape=(20, 1, 5, 5),
    #                   poolsize=(2, 2),
    #                   activation_fn=relu),
    #     FullyConnectedLayer(
    #         n_in=20 * 12 * 12, n_out=16, activation_fn=relu, p_dropout=0.5),
    #     SoftmaxLayer(n_in=16, n_out=10, p_dropout=0.5)],
    #     mini_batch_size)
    net.sgd(expanded_training_data, 30, mini_batch_size, 0.03,
            validation_data, test_data)
    return net


def train_multi_net(net_count):
    expanded_training_data, _, _ = load_data_shared(
        "../data/mnist_expanded.pkl.gz")
    file_names = get_file_name()
    for i in range(net_count):
        print 'Training net{}:'.format(i)
        train_net(expanded_training_data).save(file_names.next())
        print 'Finished Training net' + str(i)


def main():
    # train_multi_net(5)

    multi_net = MultiNet([load_net('Saved\\test_net0.txt'),
                          load_net('Saved\\test_net1.txt'),
                          load_net('Saved\\test_net2.txt'),
                          load_net('Saved\\test_net3.txt'),
                          load_net('Saved\\test_net4.txt')])
    # print load_net('Saved\\test_net0.txt').feedforward(test_by_hand())
    char = test_by_hand()
    print multi_net.feedforward(char)

    # net = neural_network.load(get_file_name())
    # test_by_hand(net)
    # activate()
    # training_data, validation_data, test_data = \
    #     mnist_loader.load_data_wrapper()
    # sizes = [784, 100, 100, 10]
    # time1 = time.time()
    # time2 = time.time()
    # print (time2 - time1) * 1000.0
    pass


if __name__ == '__main__':
    main()