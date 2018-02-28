# -*- coding: utf-8 -*-
import cPickle
import os
import numpy as np
from PIL import Image


def extract_images_from_dataset(num_of_images, dataset, mapping):
    images, labels = dataset
    with open('dataset_mapping.txt', 'w') as f:
        cPickle.dump(mapping, f)
    for index, (label, image) in enumerate(
            zip(labels[:num_of_images],
                images[:num_of_images])):
        char = chr(mapping[label])
        if not os.path.exists('dataset\\' + char):
            os.makedirs('dataset\\' + char)
        illustrate_canvas('dataset\\{}\\{}.png'.format(char, index),
                          image.reshape(28, 28))


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
    img_arr = np.zeros((canvas.shape[0], canvas.shape[1], 2), dtype=np.uint8)
    img_arr[:] = 255
    img_arr[:, :, 0] = a
    # print img_arr
    img = Image.fromarray(img_arr, 'LA')
    img.save(filename)
