# -*- coding: utf-8 -*-
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import mnist_loader
import cPickle
# from neural_network import Network
# import neural_network
# import time
from scipy import misc
import numpy as np
from PIL import Image  # , ImageOps
from scipy import ndimage
from neu_net_test import Network, ConvPoolLayer, FullyConnectedLayer, \
    SoftmaxLayer, relu, load_data_shared, MultiNet, load_net
# import preprocessing_test1


def img_to_bytes(filename):
    # with open('test_byte_stuff', 'w') as f:
    im = Image.open(filename).convert('LA')
    pixels = im.load()
    array = []
    for y in xrange(28):
        for x in xrange(28):
            intensity = 1 - (pixels[x, y][0] / 255.0)
            # f.write(str(intensity) + ' ')
            array.append(intensity)
            # if pix[x, y] == (0, 255):
            #     f.write('1')
            # else:
            #     f.write('0')
    return array


def get_file_name():
    save_dir = 'Saved Nets'
    i = 0
    while True:
        file_name = 'test_net{}.txt'.format(i)
        yield save_dir + '\\' + file_name
        i += 1


# def activate():
#     training_data, validation_data, test_data = \
#         mnist_loader.load_data_wrapper()
#     sizes = [784, 100, 100, 10]
#     net = Network(sizes)
#     learning_rate = 0.25
#     lmbda = 5
#     net.stochastic_gradient_descent(training_data, 10, learning_rate,
#                                     lmbda, test_data=test_data)
#     net.save_best(get_file_name())


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


def get_arr_from_img(filename):
    pixels = img_to_bytes(filename)
    # with open('test_byte_stuff', 'r') as f:
    #     bits = map(float, str(f.read()).split(' ')[:-1])
    # canvas = np.zeros((784, 1))
    canvas = np.array(pixels).reshape((28, 28))
    # for index, val in zip(canvas, bits):
    #     index[0] = val
    # Get bounding box of char
    min_x, min_y, max_x, max_y = get_bounding_box(canvas)
    boundbox_x = max_x - min_x + 1
    boundbox_y = max_y - min_y + 1
    # Rescale
    # scaling = int(round(20. / max(boundbox_x, boundbox_y)))  # Get scaling
    scaling = (round(20. / max(boundbox_x, boundbox_y)))  # Get scaling
    # needed
    char = canvas[min_y:max_y + 1, min_x:max_x + 1]
    # zoomed_char = np.kron(char, np.ones((scaling, scaling)))  # Rescale the
                                                              # char to  be
                                                              #  20 * 20 pixels
    zoomed_char = misc.imresize(char, scaling)
    # illustrate_canvas('testzoom.png', zoomed_char)
    # Place scaled char on blank canvas
    canvas = np.zeros((28, 28))
    canvas[:zoomed_char.shape[0], :zoomed_char.shape[1]] = zoomed_char
    # Center scaled char by it's center of mass
    center = map(round, ndimage.measurements.center_of_mass(canvas))
    shift = (28 / 2 - center[0], 28 / 2 - center[1])
    canvas = ndimage.interpolation.shift(canvas, shift).reshape((784, 1))
    return canvas
    # b = net.feedforward(a)
    # print b
    # print np.argmax(b)


def illustrate_canvas(filename, canvas):
    """
    Saves the canvas as an image. Useful for preprocessing, to determine the
    effects of each step.
    :param filename: String - The name to save the image to.
    :param canvas: np.array - The canvas to illustrate.
    """
    a = canvas
    # a = (1 - a) * 225.0
    a = 255.0 - a
    print a
    img_arr = np.zeros((canvas.shape[0], canvas.shape[1], 2), dtype=np.uint8)
    img_arr[:] = 255
    img_arr[:, :, 0] = a
    # print img_arr
    img = Image.fromarray(img_arr, 'LA')
    img.save(filename)


def train_net(training_data, test_data):  # , validation_data):
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
            n_in=1000, n_out=300, activation_fn=relu, p_dropout=0.5),
        SoftmaxLayer(n_in=300, n_out=47, p_dropout=0.5)],
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
    net.sgd(training_data, 30, mini_batch_size, 0.03, test_data)
    return net


def train_multi_net(net_count):
    # training_data, validation_data, test_data = load_data_shared()
    training_data, test_data = load_data_shared()
    # expanded_training_data, _, _ = load_data_shared(
    #     "../data/mnist_expanded.pkl.gz")
    file_names = get_file_name()
    for i in range(net_count):
        print 'Training net{}:'.format(i)
        # train_net(expanded_training_data, test_data).\
        #     save(file_names.next())
        train_net(training_data, test_data).\
            save(file_names.next())
        print 'Finished Training net' + str(i)


def test(img, boundbox):
    im = Image.open(img)
    # print im.size
    pixels = im.load()
    array = []
    for y in xrange(im.size[1]):
        for x in xrange(im.size[0]):
            intensity = (255 - pixels[x, y][0]) / 255.0
            array.append(intensity)
    array = np.array(array).reshape((im.size[1], im.size[0]))
    # print array.shape  # [boundbox[0]:boundbox[1], boundbox[2]:boundbox[3]]
    # print boundbox
    im = array[boundbox[0]:boundbox[1], boundbox[2]:boundbox[3]]
    # illustrate_canvas('testbla.png', im)

    # canvas = np.array(pixels).reshape((28, 28))
    # min_x, min_y, max_x, max_y = get_bounding_box(canvas)
    # boundbox_x = max_x - min_x + 1
    # boundbox_y = max_y - min_y + 1
    # # Rescale
    # scaling = int(round(20. / max(boundbox_x, boundbox_y)))  # Get scaling
    # # needed
    # char = canvas[min_y:max_y + 1, min_x:max_x + 1]
    # zoomed_char = np.kron(char, np.ones((scaling, scaling)))  # Rescale the
    #                                                           # char to  be
    #                                                         #  20 * 20 pixels
    # # Place scaled char on blank canvas
    canvas = np.zeros((28, 28))
    canvas[:im.shape[0], :im.shape[1]] = im
    # Center scaled char by it's center of mass
    center = map(round, ndimage.measurements.center_of_mass(canvas))
    shift = (28 / 2 - center[0], 28 / 2 - center[1])
    canvas = ndimage.interpolation.shift(canvas, shift)
    # illustrate_canvas('testbla.png', canvas)
    canvas = canvas.reshape((784, 1))
    return canvas


def main():
    # train_multi_net(1)
    with open('..\\dataset\\dataset_mapping.txt') as mapping_file:
        mapping = cPickle.load(mapping_file)
    multi_net = MultiNet([load_net('Saved Nets\\test_net1.txt')])#,
    #                       # load_net('Saved Nets\\test_net1.txt'),
    #                       # load_net('Saved Nets\\test_net2.txt'),
    #                       # load_net('Saved Nets\\test_net3.txt'),
    #                       # load_net('Saved Nets\\test_net4.txt')])
    # print load_net('Saved Nets\\test_net1.txt').feedforward(test_by_hand())
    # char = test('numbers.png', preprocessing_test1.main()[0])
    char = get_arr_from_img('test_draw.png')
    # print multi_net.feedforward(char)
    print chr(mapping[multi_net.feedforward(char)[0]])

    # char = test('numbers.png', preprocessing_test1.main()[1])
    # print multi_net.feedforward(char)
    # char = test('numbers.png', preprocessing_test1.main()[2])
    # print multi_net.feedforward(char)
    # char = test('numbers.png', preprocessing_test1.main()[3])
    # print multi_net.feedforward(char)
    # char = test('numbers.png', preprocessing_test1.main()[4])
    # print multi_net.feedforward(char)
    
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