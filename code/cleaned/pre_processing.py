# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from scipy import ndimage  # , misc
import cv2


BOUNDING_BOX_THRESHOLD = 100
CHARACTER_SIZE = 20.


class ImgSizeError(Exception):
    def __str__(self):
        return 'The size of the image is not as expected.'


def load_img_arr(filename):
    """
    Returns a numpy array of "pixel intensities" (ranging from 0 to 255) of the
    grayscaled image. Image HAS to be 28X28 pixels. Used for testing on
    individual characters.
    :param filename: The path of the image.
    :return: Numpy array of the pixel intensities.
    """
    im = Image.open(filename).convert('LA')
    if im.size != (28, 28):
        raise ImgSizeError
    pixels = im.load()
    array = [255. - pixels[x, y][0] for y in range(28) for x in range(28)]
    return np.array(array).reshape((28, 28))


def get_bounding_box(img_arr):
    """
    Returns the bounding box of the character in the image. Determined by the
    BOUNDING_BOX_THRESHOLD, which gives a minimum intensity of a pixel that is
    considered as part of the character.
    :param img_arr: 2D list or numpy array of the pixel intensities of the
    image.
    :return (min x, min y, max x, max y) of the character.
    """
    min_x = len(img_arr)
    min_y = len(img_arr[0])
    max_x = -1
    max_y = -1
    for y in range(len(img_arr)):
        for x in range(len(img_arr[0])):
            if img_arr[y][x] > BOUNDING_BOX_THRESHOLD:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    return min_x, min_y, max_x, max_y


def rescale_char_in_img_arr(img_arr):
    """
    Rescales the character in the image to the one defined by CHARACTER_SIZE,
    then puts it at the top left of the image. It will only rescale until
    either the width or the height reach the specified size, so it will keep
    the character's height/width ratio.
    :param img_arr: A numpy array of the pixel intensities of the image.
    :return: A new numpy array with a rescaled version of the character at it's
    top-left corner.
    """
    # Get bounding box
    min_x, min_y, max_x, max_y = get_bounding_box(img_arr)
    boundbox_x = max_x - min_x + 1
    boundbox_y = max_y - min_y + 1
    # Rescale
    # scaling = round(CHARACTER_SIZE / max(boundbox_x, boundbox_y))  # Get
    scaling = round(CHARACTER_SIZE / max(boundbox_x, boundbox_y))  # Get
    #scaling needed
    char = img_arr[min_y:max_y + 1, min_x:max_x + 1]
    # zoomed_char = misc.imresize(char, scaling)
    zoomed_char = cv2.resize(char, (0, 0), fx=scaling, fy=scaling)
    # Place scaled char on blank canvas
    canvas = np.zeros_like(img_arr)
    canvas[:zoomed_char.shape[0], :zoomed_char.shape[1]] = zoomed_char
    return canvas


def center_char_in_img_arr(img_arr):
    """
    Centers the character on an image to the center  # by using it's center of
    mass.
    :param img_arr: A numpy array of the pixel intensities of the image.
    :return: A new numpy array with the character centered.
    """
    height, width = img_arr.shape
    # center_of_mass = ndimage.measurements.center_of_mass(img_arr)
    # shift = (round(height / 2 - center_of_mass[0]),
    #          round(width / 2 - center_of_mass[1]))
    min_x, min_y, max_x, max_y = get_bounding_box(img_arr)
    center = ((max_y - min_y) / 2, (max_x - min_x) / 2)
    shift = (round(height / 2 - center[0]),
             round(width / 2 - center[1]))
    return ndimage.interpolation.shift(img_arr, shift)


def preprocess_img(filename):
    """
    Rescales and centers an image to fit the MNIST conventions.
    :param filename: The image path
    :return: A numpy array
    """
    pixels = load_img_arr(filename)
    canvas = np.array(pixels)
    canvas = rescale_char_in_img_arr(canvas)
    canvas = center_char_in_img_arr(canvas)
    return canvas / 255.


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
