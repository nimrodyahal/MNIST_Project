"""
neural_network.py
~~~~~~~~~~~~~~
Repurposed code - credit goes to Michael Nielsen. Original code can be found at
https://github.com/mnielsen/neural-networks-and-deep-learning.
It is HIGHLY recommended to read extensively on the subject before reading the
code, as even with the documentation, it cannot be understood without the
considerable theoretical knowledge.

A Theano-based program for training and running simple neural networks.
Supports several layer types (fully connected, convolutional, max
pooling, softmax).
Can run on CPU, or GPU, with GPU being significantly faster.
Written for Theano 0.6 and 0.7, needs some changes for more recent
versions of Theano.
"""

import cPickle
import scipy.io as sio
import numpy as np
import theano
import theano.tensor as tt
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import pool
from theano.tensor.nnet import sigmoid


# PATH = '..\\..\\dataset\\matlab\\emnist-bymerge.mat'
PATH = 'D:\\School\\Programming\\Cyber\\FinalExercise-12th\\MNIST_Project\\' \
       'dataset\\matlab\\emnist-bymerge.mat'
# NAMES = {'bal': 'emnist-balanced.mat', 'cls': 'emnist-byclass.mat',
#          'mrg': 'emnist-bymerge.mat', 'dgt': 'emnist-digits.mat',
#          'ltr': 'emnist-letters.mat', 'mnist': 'emnist-mnist'}
NUM_OF_TRIES = 5  # The amount of 'tries' a net has to classify


# Activation functions for neurons
def relu(z):
    return tt.maximum(0.0, z)


def shared(data):
    """
    Place the data into shared variables.  This allows Theano to copy
    the data to the GPU, if one is available.
    """
    shared_x = theano.shared(
        np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(
        np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
    return shared_x, tt.cast(shared_y, 'int32')


def load_data_shared():  # dataset='mrg'):
    """
    Loads data from dataset
    :param: dataset: The specific dataset to load. Options are:
        'bal' - Balanced
        'cls' - By_Class
        'mrg' - By_Merge
        'dgt' - Digits
        'ltr' - Letters
        'mnist' - MNIST
    :return: [training_images, training_labels],\
           [testing_images, testing_labels], mapping
    """
    width = 28
    height = 28
    mat = sio.loadmat(PATH)  # + NAMES[dataset])

    training_len = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0]. \
        reshape(training_len, height, width, 1)
    training_labels = mat['dataset'][0][0][0][0][0][1]
    training_labels = [np.int64(label[0]) for label in training_labels]

    testing_len = len(mat['dataset'][0][0][1][0][0][0])
    testing_images = mat['dataset'][0][0][1][0][0][0]. \
        reshape(testing_len, height, width, 1)
    testing_labels = mat['dataset'][0][0][1][0][0][1]
    testing_labels = [np.int64(label[0]) for label in testing_labels]

    for i in range(len(training_images)):
        training_images[i] = np.rot90(np.fliplr(training_images[i]))
    for i in range(len(testing_images)):
        testing_images[i] = np.rot90(np.fliplr(testing_images[i]))

    training_images = training_images.reshape(training_len, width * height)
    testing_images = testing_images.reshape(testing_len, width * height)

    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    training_images /= 255
    testing_images /= 255

    training_data = [training_images, training_labels]
    testing_data = [testing_images, testing_labels]

    mapping = {kv[0]: kv[1:][0] for kv in mat['dataset'][0][0][2]}
    return shared(training_data), shared(testing_data), mapping


#### Main class used to construct and train networks
class Network(object):
    def __init__(self, layers, mini_batch_size, mapping):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.
        """
        self.mapping = mapping
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in
                       layer.params]
        self.x = tt.matrix('x')
        self.y = tt.ivector('y')
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer = self.layers[j - 1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout,
                self.mini_batch_size)

    def __classify_word(self, inpt, starting_point, char_count):
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, char_count)
        for x in xrange(1, len(self.layers)):
            prev_layer, layer = self.layers[x - 1], self.layers[x]
            layer.set_inpt(prev_layer.output, prev_layer.output_dropout,
                           char_count)

        starting_point_tensor = tt.lscalar()
        char_count_tensor = tt.lscalar()
        feedforward_function = theano.function(
            [starting_point_tensor, char_count_tensor], self.layers[-1].output,
            givens={self.x: inpt[
                    starting_point_tensor:
                    starting_point_tensor + char_count_tensor]})
        return feedforward_function(starting_point, char_count)

    @staticmethod
    def __lines_to_chars(lines):
        chars = np.zeros((1, 28, 28))
        for line in lines:
            for word in line:
                for char in word:
                    chars = np.append(chars, [char], axis=0)
        return chars[1:]

    # def classify_text(self, input_lines):
    #     chars = self.__lines_to_chars(input_lines)
    #     inpt = np.array(chars)
    #     inpt = inpt.reshape((-1, 28 * 28))
    #     print inpt.shape
    #     inpt = theano.shared(
    #         np.asarray(inpt, dtype=theano.config.floatX), borrow=True)
    #
    #     starting_point = 0
    #     classified_text = []
    #     for line in input_lines:
    #         classified_line = []
    #         for word in line:
    #             char_count = len(word)
    #             print starting_point, char_count
    #             classified_line.append(self.__classify_word(
    #                 inpt, starting_point, char_count))
    #             starting_point += char_count
    #         classified_text.append(classified_line)
    #     return classified_text

    def classify_text(self, input_lines):
        chars = self.__lines_to_chars(input_lines)
        inpt = np.array(chars).reshape((-1, 28 * 28))
        inpt = theano.shared(
            np.asarray(inpt, dtype=theano.config.floatX), borrow=True)

        char_count = len(chars)
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, char_count)
        for x in xrange(1, len(self.layers)):
            prev_layer, layer = self.layers[x - 1], self.layers[x]
            layer.set_inpt(prev_layer.output, prev_layer.output_dropout,
                           char_count)

        i = tt.lscalar()
        feedforward_function = theano.function(
            [i], self.layers[-1].output,
            givens={self.x: inpt}, on_unused_input='ignore')

        classified_chars = feedforward_function(0)
        classified_text = []
        for l_index in range(len(input_lines)):
            classified_line = []
            for w_index in range(len(input_lines[l_index])):
                classified_word = []
                for c_index in range((len(input_lines[l_index][w_index]))):
                    classified_word.append(classified_chars[0])
                    classified_chars = classified_chars[1:]
                classified_line.append(classified_word)
            classified_text.append(classified_line)
        return classified_text

    def sgd(self, training_data, epochs, mini_batch_size, learning_rate,
            test_data, lmbda=0.0):
        """
        Train the network using mini-batch stochastic gradient descent.
        :param training_data: (theano shared numpy array - input, theano shared
        numpy array - output) - The training examples.
        :param epochs: int - The number of epochs to run.
        :param mini_batch_size: int - The mini batch size.
        :param learning_rate: int - The learning rate.
        :param test_data: (theano shared numpy array - input, theano shared
        numpy array - output) - The testing examples.
        :param lmbda: int - Used in the normalization of the cost function.
        """
        training_input, training_output = training_data
        test_input, test_output = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data) / self.mini_batch_size
        num_test_batches = size(test_data) / self.mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and
        # updates
        l2_norm_squared = sum([(layer.w ** 2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) + \
            0.5 * lmbda * l2_norm_squared / num_training_batches
        grads = tt.grad(cost, self.params)
        updates = [(param, param - learning_rate * grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = tt.lscalar()  # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x: training_input[
                    i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y: training_output[
                    i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x: test_input[
                    i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y: test_output[
                    i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            }, on_unused_input='warn')
        # Do the actual training
        best_test_accuracy = 0.0
        best_iteration = 0
        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches * epoch + minibatch_index
                if iteration % 1000 == 0:
                    print('Training mini-batch number {0} ({1:.2%})'.format
                          (iteration,
                           float(minibatch_index)/num_training_batches))
                train_mb(minibatch_index)
                if (iteration + 1) % num_training_batches == 0:
                    test_accuracy = np.mean(
                        [test_mb_accuracy(j) for j in
                         xrange(num_test_batches)])
                    print 'Epoch {0}: test accuracy {1:.2%}' \
                        .format(epoch, test_accuracy)
                    if test_accuracy >= best_test_accuracy:
                        print 'This is the best test accuracy to date.'
                        best_test_accuracy = test_accuracy
                        best_iteration = iteration
                    print 'Best test accuracy is {0:.2%} on epoch {1}' \
                        .format(test_accuracy, epoch)
        print 'Finished training network.'
        print 'Best test accuracy of {0:.2%} obtained at iteration {1}' \
            .format(best_test_accuracy, best_iteration)

    def save(self, filename):
        with open(filename, 'wb') as f:
            cPickle.dump(self, f, -1)


#### Define layer types
class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.
    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and
        the filter width.
        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.
        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                 np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0 / n_out),
                                 size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]
        self.inpt = self.output = self.output_dropout = None

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size=None):
        if not mini_batch_size:
            self.inpt = inpt.reshape(self.image_shape)
            conv_out = conv2d(
                input=self.inpt, filters=self.w,
                filter_shape=self.filter_shape, image_shape=self.image_shape)
        else:
            new_shape = list(self.image_shape)
            new_shape[0] = mini_batch_size
            self.inpt = inpt.reshape(new_shape)
            conv_out = conv2d(
                input=self.inpt, filters=self.w,
                filter_shape=self.filter_shape, input_shape=new_shape)
        pooled_out = pool.pool_2d(
            input=conv_out, ws=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output  # no dropout in the convolutional
        # layers


class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]
        self.inpt = self.output = self.output_dropout = self.y_out = \
            self.inpt_dropout = None

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1 - self.p_dropout) * tt.dot(self.inpt, self.w) + self.b)
        self.y_out = tt.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            tt.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        """Return the accuracy for the mini-batch."""
        return tt.mean(tt.eq(y, self.y_out))


class SoftmaxLayer(object):
    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]
        self.inpt = self.output = self.output_dropout = self.y_out = \
            self.inpt_dropout = None

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = (1 - self.p_dropout) * tt.dot(self.inpt, self.w) + self.b
        self.y_out = tt.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(
            tt.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        """
        Return the log-likelihood cost.
        """
        return -tt.mean(
            tt.log(self.output_dropout)[tt.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        """
        Return the accuracy for the mini-batch. Takes the first 3 predictions
        of the net into account.
        """
        # Get the keys sorted by values in descending order.
        likelihoods = self.output.argsort(axis=1)[:, ::-1]
        # Check if the predictions match the reality. Uses logic gate 'or' to
        #  include all tries.
        prediction = tt.eq(y, likelihoods[:, 0])  # For some reason tt.zeros_
        #  like (which would have made a tensor of zeros the shape of y)
        #  doesn't work, so I used this method instead.
        for i in range(1, NUM_OF_TRIES):
            prediction = tt.or_(prediction, tt.eq(y, likelihoods[:, i]))
        return tt.mean(prediction)


class MultiNet():
    def __init__(self, nets):
        self.nets = nets

    def classify_text(self, lines):
        mapping = self.nets[0].mapping
        answer = [net.classify_text(lines) for net in self.nets]

        classified_text = []
        for line in np.array(answer)[0]:
            classified_line = []
            for word in line:
                classified_word = []
                for char in word:
                    classifications = char.argsort()[::-1][:NUM_OF_TRIES]
                    classified_word.append([(chr(mapping[c]), char[c]) for c
                                            in classifications])
                classified_line.append(classified_word)
            classified_text.append(classified_line)
        return classified_text


#### Miscellanea
def size(data):
    """Return the size of the dataset `data`."""
    return data[0].get_value(borrow=True).shape[0]


def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1 - p_dropout, size=layer.shape)
    return layer * tt.cast(mask, theano.config.floatX)


# def load_net(filename):
#     with open(filename, 'rb') as f:
#         return cPickle.load(f)