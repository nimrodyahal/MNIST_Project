# -*- coding: utf-8 -*-
from scipy import misc
import numpy as np
from PIL import Image
from scipy import ndimage


BOUNDING_BOX_THRESHOLD = 0.01


def img_to_bytes(filename):
    im = Image.open(filename).convert('LA')
    pixels = im.load()
    array = []
    for y in xrange(28):
        for x in xrange(28):
            # intensity = 1 - (pixels[x, y][0] / 255.0)
            intensity = 255.0 - pixels[x, y][0]
            array.append(intensity)
    return array


def get_bounding_box(img):
    min_x = len(img)
    min_y = len(img[0])
    max_x = -1
    max_y = -1
    for y in range(len(img)):
        for x in range(len(img[0])):
            if img[y][x] > BOUNDING_BOX_THRESHOLD:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    return min_x, min_y, max_x, max_y


def load_img_arr(filename):
    pixels = img_to_bytes(filename)
    canvas = np.array(pixels).reshape((28, 28))
    min_x, min_y, max_x, max_y = get_bounding_box(canvas)
    boundbox_x = max_x - min_x + 1
    boundbox_y = max_y - min_y + 1
    # Rescale
    scaling = (round(20. / max(boundbox_x, boundbox_y)))  # Get scaling needed
    char = canvas[min_y:max_y + 1, min_x:max_x + 1]
    zoomed_char = misc.imresize(char, scaling)
    # Place scaled char on blank canvas
    canvas = np.zeros((28, 28))
    canvas[:zoomed_char.shape[0], :zoomed_char.shape[1]] = zoomed_char
    # Center scaled char by it's center of mass
    center = map(round, ndimage.measurements.center_of_mass(canvas))
    shift = (28 / 2 - center[0], 28 / 2 - center[1])
    canvas = ndimage.interpolation.shift(canvas, shift)
    canvas = canvas.reshape((784, 1))
    return canvas


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
