# -*- coding: utf-8 -*-
import cv2
import numpy as np
import Image


# IMG_THRESHOLD = 90
SPACE_RATIO = 1.2
# SPACE_RATIO = 3
LINE_THRESHOLD = 30
MIN_CNT_HEIGHT_LINE_RATIO = 2
MIN_CNT_SIZE = 500

NET_INPUT_WIDTH = 28
NET_INPUT_HEIGHT = 28
NETWORK_INPUT_SIZE = (NET_INPUT_HEIGHT, NET_INPUT_WIDTH)
CHARACTER_SIZE = 20.0
BORDER_SIZE = 1, 1, 1, 1


class Preprocessor():
    def __init__(self, img):
        self.__img = self.__prepare_img(img)

        _, self.__contours, self.__hierarchy = cv2.findContours(
            self.__img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.__contours = np.array(self.__contours)

    def __prepare_img(self, img):
        """
        Returns a rotated to text tilt threshold of the image. Uses
        IMG_IMG_THRESHOLD to determine the threshold value.
        :param img: The image in question (OpenCV2 image).
        :return: A rotated to text tilt threshold of the image (OpenCV2 image).
        """
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # thresh = cv2.threshold(img_gray, IMG_THRESHOLD, 255,
        #                        cv2.THRESH_BINARY)[1]
        # thresh = cv2.bitwise_not(thresh)

        # thresh = cv2.adaptiveThreshold(img, 255,
        #                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                cv2.THRESH_BINARY, 41, 20)

        # blur = cv2.GaussianBlur(img, (1, 1), 0)
        # ret2, thresh = cv2.threshold(img, 0, 255,
        #                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        blur = cv2.bilateralFilter(img, 40, 10, 10)
        thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 201, 30)

        thresh = cv2.bitwise_not(thresh)
        thresh = self.__clean_image(thresh)

        # cv2.imshow('thresh', cv2.resize(thresh, (0, 0), fx=.5, fy=.5))
        # cv2.waitKey(0)
        angle = self.__find_text_tilt(thresh)
        rotated = self.__rotate_image(thresh, angle)
        # cv2.imshow('rotated', cv2.resize(rotated, (0, 0), fx=.5, fy=.5))
        # cv2.waitKey(0)
        return rotated

    @staticmethod
    def __clean_image(img):
        """
        Cleans the image from tiny dots leftover from the threshold.
        :param img: The image in question (OpenCV2 Image)
        """
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[1]
        sizes = [cv2.boundingRect(cnt)[2:] for cnt in contours]
        sizes = [(w - 1) * (h - 1) for w, h in sizes]
        for i, size in enumerate(sizes):
            if size < MIN_CNT_SIZE:
                cv2.drawContours(img, contours, i, 0, -1)

        # cv2.imshow('thresh', cv2.resize(img, (0, 0), fx=1, fy=1))
        # cv2.waitKey(0)
        return img
        # for i, (w, h) in enumerate(sizes):
        #     if w < MIN_CNT_SIZE and h < MIN_CNT_SIZE:
        #         cv2.drawContours(img, contours, i, 0, -1)

    @staticmethod
    def __find_text_tilt(img):
        """
        Returns the angle of the text tilt, so that it can be corrected.
        Uses OpenCV2's minAreaRect to find the text area, and from that, the
        angle of the text.
        :param img: The threshold image of the image with the text.
        :return: The angle of the text tilt.
        """
        coords = np.column_stack(np.where(img > 0))
        angle = abs(cv2.minAreaRect(coords)[-1])
        if angle > 45:
            angle -= 90
        return angle

    @staticmethod
    def __get_line_coords(img, threshold):
        """
        Returns the y coordinates of the line separations in the image.
        NOTE: The coordinates are in the middle of each line separation
        (includes the first and last lines by treating the top and bottom of
        the image as the end and beginning of lines respectively).
        :param img: The image in question (OpenCV2 image).
        :param threshold: The minimum number of empty lines to count as a line
        separation.
        :return: [(line_starts, line_ends)] for each line.
        """
        slices = cv2.reduce(img, 1, cv2.REDUCE_AVG).reshape(-1)
        slice_threshold = np.mean(slices) / 4
        count = 0
        is_line = False
        line_coords = []
        for i, slc in enumerate(slices):
            if is_line:
                if not slc >= slice_threshold:
                    is_line = False
                    count = 1
            else:
                if slc >= slice_threshold:
                    is_line = True
                    if count >= threshold:
                        line_coords.append(i - count / 2)
                else:
                    count += 1
        if not line_coords:
            line_coords.append(0)
        line_coords.append(img.shape[0])
        return zip(line_coords[:-1], line_coords[1:])

    # def __get_line_coords(self):
    #     contours = cv2.findContours(self.__img, cv2.RETR_EXTERNAL,
    #                                 cv2.CHAIN_APPROX_SIMPLE)[1]
    #     sizes = [(cv2.boundingRect(cnt)[1], cv2.boundingRect(cnt)[3])
    #              for cnt in contours]
    #     sizes = np.array(sorted(sizes, key=lambda x: x[0]))
    #     line_breaks = []
    #     for (y1, h1), (y2, h2) in zip(sizes[:-1], sizes[1:]):
    #         overlap = (y1 + h1) - y2
    #         if overlap < 0 or h2 / overlap > 3:
    #             line_break = (y2 + (y1 + h1)) / 2
    #             line_breaks.append(line_break)
    #     lines = [0] + line_breaks + [self.__img.shape[0]]
    #     return zip(lines[:-1], lines[1:])

    @staticmethod
    def __rotate_image(img, angle):
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

    @staticmethod
    def __get_line_height(line):
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

    def __get_word_coords(self, line, threshold):
        """
        Returns the x coordinates of the word separations in the line.
        NOTE: The coordinates are in the middle of each word separation
        (includes the first and last word by treating the left and right of the
        image as the end and beginning of words respectively).
        :param line: The line in question (OpenCV2 image).
        :param threshold: The minimum number of empty spaces to count as a word
        separation.
        :return: [(word_starts, word_ends)] for each word. (int, int)
        """
        rotated = cv2.rotate(line, cv2.ROTATE_90_CLOCKWISE)
        return self.__get_line_coords(rotated, threshold)

    @staticmethod
    def __get_padding(char):
        """
        Returns how much padding is needed to turn the character to the
        specified input shape of the neural network.
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

    def __rescale_char(self, char):
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

        pad_t, pad_b, pad_l, pad_r = self.__get_padding(rescaled)
        rescaled = cv2.copyMakeBorder(rescaled, pad_t, pad_b, pad_l, pad_r,
                                      cv2.BORDER_CONSTANT, value=black)
        return rescaled

    def __crop_contour(self, contours):
        """
        Redraws the character so that only the relevant parts remain.
        :param contours: List of all the contours that make the character.
        :return: The redrawn character (OpenCV2 image).
        """
        mask = np.zeros(self.__img.shape)
        cv2.drawContours(mask, contours, -1, 255, -1)
        # out = np.zeros(self.__img.shape)
        # out[mask == 255] = mask[mask == 255]

        # _time0 = time()
        out = mask
        # # Now crop
        x, y = np.where(mask == 255)
        # print time() - _time0
        #
        top_x, top_y = (np.min(x), np.min(y))
        bottom_x, bottom_y = (np.max(x), np.max(y))

        # print 'len', len(contours)
        # x, y, w, h = cv2.boundingRect(contours[0])
        # top_y2, top_x2 = y, x
        # bottom_y2, bottom_x2 = y + h, x + w
        # print'top_x:', top_x, top_x2
        # print'top_y:', top_y, top_y2
        # print'bottom_x:', bottom_x, bottom_x2
        # print'bottom_y:', bottom_y, bottom_y2

        return out[top_x:bottom_x+1, top_y:bottom_y+1]

    # def tittle_segmentation(self, upper, lower):
    #     for i, cnt in enumerate(self.__contours):
    #         bounds = (upper, lower, 0, self.__img.shape[1])
    #         if self.__filter_contour(i, 0, bounds):
    #             pass
    #     pass

    def __filter_contour(self, i, line_height, bounds):
        """
        Returns whether the contour is a character inside the word or not.
        Checks that the contour is an external one, if it's inside the bounds
        of the word, and if it is tall enough to not be a tittle (the little
        dots above 'i' and 'j').
        :param i: The index of the contour.
        :param line_height: The average height of the contours in the word.
        :param bounds: (upper, lower, left, right).
        :return: True if the contour is a character in the word. False
        otherwise.
        """
        upper, lower, left, right = bounds
        if self.__hierarchy[0, i, 3] != -1:
            return False
        x, y, w, h = cv2.boundingRect(self.__contours[i])
        x_center = x + w / 2
        y_center = y + h / 2
        if y_center <= upper or y_center >= lower or \
                x_center <= left or x_center >= right:
            return False
        if h + 2 < line_height / MIN_CNT_HEIGHT_LINE_RATIO:
            return False
        return True

    def __separate_word(self, word_coords, line_height):
        """
        Separate each word to its characters. (Also turns character to network
        format of 0-1 instead of 0-255.)
        :param line_height: The height of the line, used to filter contours
        that are too small.
        :return: List of characters.
        :rtype : [OpenCV2 image]
        """
        word_chars = []
        for i, cnt in enumerate(self.__contours):
            if self.__filter_contour(i, line_height, word_coords):
                hie = list(np.where(self.__hierarchy[0, :, 3] == i)[0]) + [i]
                char = self.__crop_contour(self.__contours[hie])
                x = cv2.boundingRect(cnt)[0]
                word_chars.append((char / 255.0, x))
        word_chars = [w[0] for w in sorted(word_chars, key=lambda a: a[1])]
        return word_chars

    def __separate_line(self, img_line, line_coords):
        """
        Separate each line to its characters.
        :param img_line: The line in question (OpenCV2 image).
        :return: List of lists of characters (OpenCV2 image).
        :rtype : [words] = [[(OpenCV2 image)]]
        """
        upper, lower = line_coords
        line_height = self.__get_line_height(img_line)
        space_len = line_height / SPACE_RATIO

        words = self.__get_word_coords(img_line, space_len)
        line = []
        for left, right in words:
            chars = self.__separate_word((upper, lower, left, right),
                                         line_height)
            word_chars = [self.__rescale_char(char) for char in chars]
            if word_chars:
                line.append(word_chars)
        return line

    def separate_text(self):
        """
        Separates the image to its individual characters, ready to be inputted
        to the neural network.
        :return: List of lists of lists of characters (OpenCV2 image).
        :rtype : [lines] = [[words]] = [[[(OpenCV2 image)]]]
        """
        _, self.__contours, self.__hierarchy = cv2.findContours(
            self.__img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.__contours = np.array(self.__contours)

        lines = self.__get_line_coords(self.__img, LINE_THRESHOLD)
        text = []
        for upper, lower in lines:
            img_line = self.__img[upper:lower]
            line_chars = self.__separate_line(img_line, (upper, lower))
            if line_chars:
                text.append(line_chars)
        return text


# _path = '..\\testing images\\TEST3.jpg'
# _img = cv2.imread(_path, 0)
# prepr = Preprocessor(_img)
#
# _text = prepr.separate_text()
# print 'lines: ', len(_text)
# for _line in _text:
#     print '\twords: ', len(_line)
#     for _word in _line:
#         print '\t\tchars: ', len(_word)
#         # for _i, _char in enumerate(_word):
#         #     cv2.imshow(str(_i), _char * 255.0)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()