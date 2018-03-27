# -*- coding: utf-8 -*-
import cv2
import numpy as np
import Image


IMG_THRESHOLD = 100
SPACE_RATIO = 1.2
LINE_THRESHOLD = 1
MIN_CNT_HEIGHT_LINE_RATIO = 2
MIN_CNT_SIZE = 5

NET_INPUT_WIDTH = 28
NET_INPUT_HEIGHT = 28
NETWORK_INPUT_SIZE = (NET_INPUT_HEIGHT, NET_INPUT_WIDTH)
CHARACTER_SIZE = 20.0
BORDER_SIZE = 1, 1, 1, 1


def _prepare_img(img):
    """
    Returns a rotated to text tilt threshold of the image. Uses IMG_THRESHOLD
    to determine the threshold value.
    :param img: The image in question (OpenCV2 image).
    :return: A rotated to text tilt threshold of the image (OpenCV2 image).
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img_gray, IMG_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.bitwise_not(thresh)
    _clean_image(thresh)

    angle = _find_text_tilt(thresh)
    rotated = _rotate_image(thresh, angle)
    return rotated


def _clean_image(img):
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[1]
    sizes = [cv2.boundingRect(cnt)[2:] for cnt in contours]
    for i, (w, h) in enumerate(sizes):
        if w < MIN_CNT_SIZE and h < MIN_CNT_SIZE:
            cv2.drawContours(img, contours, i, 0, -1)


def _find_text_tilt(img):
    """
    Returns the angle of the text tilt, so that it can be corrected.
    Uses OpenCV2's minAreaRect to find the text area, and from that, the angle
    of the text.
    :param img: The threshold image of the image with the text.
    :return: The angle of the text tilt.
    """
    coords = np.column_stack(np.where(img > 0))
    angle = abs(cv2.minAreaRect(coords)[-1])
    if angle > 45:
        angle -= 90
    return angle


def _rotate_image(img, angle):
    """
    Rotates image to the angle specified.
    :param img: The image in question (OpenCV2 image).
    :param angle: The angle in question.
    :return: The rotated image (OpenCV2 image).
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _get_line_coords(img, threshold):
    """
    Returns the y coordinates of the line separations in the image.
    NOTE: The coordinates are in the middle of each line separation (includes
    the first and last lines by treating the top and bottom of the image as the
    end and beginning of lines respectively).
    :param img: The image in question (OpenCV2 image).
    :param threshold: The minimum number of empty lines to count as a line
    separation.
    :return: [(line_starts, line_ends)] for each line.
    """
    slices = cv2.reduce(img, 1, cv2.REDUCE_AVG).reshape(-1)
    # slice_threshold = np.mean(slices) / 4
    # print slice_threshold
    # print 'yolo'
    count = 0
    is_line = False
    line_coords = []
    for i, slc in enumerate(slices):
        if is_line:
            if not slc:
                is_line = False
                count = 1
        else:
            if slc:
                is_line = True
                if count >= threshold:
                    line_coords.append(i - count / 2)
            else:
                count += 1
    line_coords.append(img.shape[:2][0])
    return zip(line_coords[:-1], line_coords[1:])


def _get_line_height(line):
    """
    Returns the actual height of the line, by averaging the heights of each
    character in the line.
    :param line: The line in question (OpenCV2 image).
    :return:
    """
    contours = cv2.findContours(line, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[1]
    heights = [cv2.boundingRect(cnt)[3] for cnt in contours]
    return np.mean(heights)


def _get_word_coords(line, threshold):
    """
    Returns the x coordinates of the word separations in the line.
    NOTE: The coordinates are in the middle of each word separation (includes
    the first and last word by treating the left and right of the image as the
    end and beginning of words respectively).
    :param line: The line in question (OpenCV2 image).
    :param threshold: The minimum number of empty spaces to count as a word
    separation.
    :return: [(word_starts, word_ends)] for each word. (int, int)
    """
    rotated = cv2.rotate(line, cv2.ROTATE_90_CLOCKWISE)
    return _get_line_coords(rotated, threshold)


def _get_padding(char):
    """
    Returns how much padding is needed to turn the character to the specified
    input shape of the neural network.
    NOTE: Will return negative if the character is larger than the network
    input shape.
    :param char: The character in question (OpenCV2 image).
    :return: (padding_height, padding_width) for the character. (int, int)
    """
    input_h, input_w = NETWORK_INPUT_SIZE
    char_h, char_w = char.shape[:2]
    padding_left = (input_w - char_w) / 2
    padding_right = input_w - (char_w + padding_left)
    padding_top = (input_h - char_h) / 2
    padding_bottom = input_h - (char_h + padding_top)
    return padding_top, padding_bottom, padding_left, padding_right


def _rescale_char(char):
    """
    Rescales the character so that either its height or width is equal to
    CHARACTER_SIZE, then pads it so that it fits with NETWORK_INPUT_SIZE
    :param char: The character in question (OpenCV2 image).
    :return: The rescaled, padded character in question (OpenCV2 image).
    """
    bs_t, bs_b, bs_l, bs_r = BORDER_SIZE
    black = [0, 0, 0]
    char = cv2.copyMakeBorder(char, bs_t, bs_b, bs_l, bs_r,
                              cv2.BORDER_CONSTANT, value=black)

    char = char.astype(np.uint8)
    cv2_im = cv2.cvtColor(char, cv2.COLOR_GRAY2BGR)
    pil_img = Image.fromarray(cv2_im)

    w, h = pil_img.size
    scaling = CHARACTER_SIZE / max(w, h)
    new_size = (int(scaling * w), int(scaling * h))

    pil_img = pil_img.resize(new_size, Image.ANTIALIAS)
    rescaled = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)

    padding_t, padding_b, padding_l, padding_r = _get_padding(rescaled)
    rescaled = cv2.copyMakeBorder(rescaled, padding_t, padding_b, padding_l,
                                  padding_r, cv2.BORDER_CONSTANT, value=black)
    return rescaled


def _crop_contour(contours, shape):
    """
    Redraws the character so that only the relevant parts remain.
    :param contours: All the contours that make the character.
    :param shape: The shape of the word.
    :return: The redrawn character (OpenCV2 image).
    """
    mask = np.zeros(shape)
    cv2.drawContours(mask, contours, -1, 255, -1)
    out = np.zeros(shape)
    out[mask == 255] = mask[mask == 255]
    # Now crop
    x, y = np.where(mask == 255)
    top_x, top_y = (np.min(x), np.min(y))
    bottom_x, bottom_y = (np.max(x), np.max(y))
    return out[top_x:bottom_x+1, top_y:bottom_y+1]


def _separate_word(word_coords, contours, hierarchy, line_height, img_shape):
    """
    Separate each word to its characters. (Also turns character to network
    format of 0-1 instead of 0-255.)
    :param img_word: The word in question (OpenCV2 image).
    :param line_height: The height of the line, used to filter contours that
    are too small.
    :return: List of characters.
    :rtype : [OpenCV2 image]
    """
    word_chars = []
    for i, cnt in enumerate(contours):
        if _filter_contour(contours, i, hierarchy, line_height, word_coords):
            # upper, lower, left, right = word_coords
            # word_h = lower - upper
            # word_w = right - left

            hie = list(np.where(hierarchy[0, :, 3] == i)[0]) + [i]
            char = _crop_contour(contours[hie], img_shape)
            x = cv2.boundingRect(cnt)[0]
            word_chars.append((char / 255.0, x))
            # pass
        # if hierarchy[0, i, 3] == -1:
        #     h = cv2.boundingRect(cnt)[3]
        #     if h + 2 >= line_height / MIN_CNT_HEIGHT_LINE_RATIO:
    word_chars = [w[0] for w in sorted(word_chars, key=lambda a: a[1])]
    return word_chars


def _filter_contour(contours, i, hierarchy, line_height, word_coords):
    upper, lower, left, right = word_coords
    if hierarchy[0, i, 3] != -1:
        return False
    x, y, w, h = cv2.boundingRect(contours[i])
    if y <= upper or y >= lower or x <= left or x >= right:
        return False
    if h + 2 < line_height / MIN_CNT_HEIGHT_LINE_RATIO:
        return False
    return True


def _separate_line(img_line, line_coords, contours, hierarchy, img_shape):
    """
    Separate each line to its characters.
    :param img_line: The line in question (OpenCV2 image).
    :return: List of lists of characters (OpenCV2 image).
    :rtype : [words] = [[(OpenCV2 image)]]
    """
    upper, lower = line_coords
    line_height = _get_line_height(img_line)
    space_len = line_height / SPACE_RATIO
    words = _get_word_coords(img_line, space_len)
    line = []
    for left, right in words:
        chars = _separate_word((upper, lower, left, right), contours,
                               hierarchy, line_height, img_shape)
        word = [_rescale_char(char) for char in chars]
        line.append(word)
    return line


def separate_text(img):
    """
    Separates the image to its individual characters, ready to be inputted to
    the neural network.
    :param img: The image in question (OpenCV2 image).
    :return: List of lists of lists of characters (OpenCV2 image).
    :rtype : [lines] = [[words]] = [[[(OpenCV2 image)]]]
    """
    thresh = _prepare_img(img)

    _, contours, hierarchy = cv2.findContours(thresh,
                                              cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours)

    lines = _get_line_coords(thresh, LINE_THRESHOLD)
    text = []
    for upper, lower in lines:
        img_line = thresh[upper:lower]
        line_chars = _separate_line(img_line, (upper, lower), contours,
                                    hierarchy, thresh.shape)
        text.append(line_chars)
    return text

_path = '..\\testing images\\text test.jpg'
_img = cv2.imread(_path, 1)
_text = separate_text(_img)
print 'lines: ', len(_text)
for _line in _text[:3]:
    print 'words: ', len(_line)
    for _word in _line:
        print 'chars: ', len(_word)
        for _i, _char in enumerate(_word):
            cv2.imshow(str(_i), _char * 255.0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
