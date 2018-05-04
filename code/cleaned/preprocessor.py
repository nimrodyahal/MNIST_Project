# -*- coding: utf-8 -*-
import cv2
import numpy as np
import Image
from ContourClass import Contour


MIN_CNT_SIZE_RATIO = 0.0000005
Y_DISTANCE_TO_AVE_RATIO = 0.4

NET_INPUT_WIDTH = 28
NET_INPUT_HEIGHT = 28
NETWORK_INPUT_SIZE = (NET_INPUT_HEIGHT, NET_INPUT_WIDTH)
CHARACTER_SIZE = 20.0
BORDER_SIZE = 1, 1, 1, 1


class Preprocessor():
    def __init__(self, img):
        self.__img = self.__prepare_img(img)
        _, cnts, hierarchy = cv2.findContours(
            self.__img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.__contours = []
        for i, cnt in enumerate(cnts):
            self.__contours.append(Contour(cnt, i, hierarchy[0][i]))
        self.__contours = np.array(self.__contours)

    def __prepare_img(self, img):
        """
        Returns a rotated to text tilt threshold of the image. Uses
        IMG_IMG_THRESHOLD to determine the threshold value.
        :param img: The image in question (OpenCV2 image).
        :return: A rotated to text tilt threshold of the image (OpenCV2 image).
        """
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = gray
        blur = cv2.bilateralFilter(img, 40, 10, 10)
        thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 201, 30)

        thresh = cv2.bitwise_not(thresh)
        thresh = self.__clean_image(thresh)
        return thresh

    def __get_external_contours(self):
        """
        Returns only the external contours in the image.
        """
        externals = []
        for cnt in self.__contours:
            if cnt.hierarchy[3] == -1:
                externals.append(cnt)
        return externals

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
            if size < MIN_CNT_SIZE_RATIO * img.shape[0] * img.shape[1]:
                cv2.drawContours(img, contours, i, 0, -1)
        return img

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

    def __get_line_coords(self):
        """
        Returns the coordinates of the different lines.
        :returns: [(upper_y_coord, lower_y_coord)]
        """
        external = self.__get_external_contours()
        line_breaks = []
        external = sorted(external, key=lambda x: x.bounding_rect[1])
        for cnt1, cnt2 in zip(external[:-1], external[1:]):
            x1, y1, w1, h1 = cnt1.bounding_rect
            x2, y2, w2, h2 = cnt2.bounding_rect
            overlap = (y1 + h1) - y2
            if overlap <= 0 or h2 / overlap > 3:
                line_break = (y2 + (y1 + h1)) / 2
                line_breaks.append(line_break)

        lines = [0] + line_breaks + [self.__img.shape[0]]
        return zip(lines[:-1], lines[1:])

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

    def __get_word_coords(self, upper, lower):
        """
        Returns the coordinates of the different words.
        :param upper: The upper coordinate of the line.
        :param lower: The lower coordinate of the line.
        :returns: [(left_x_coord, right_x_coord)]
        """
        bounds = upper, lower, 0, self.__img.shape[1]
        external = self.__get_external_contours()  # Get only the external
        # contours
        relevant = [cnt for cnt in external if
                    self.__filter_contour(cnt, bounds)]  # Get only the
                    # relevant contours to the line

        bounding_rects = [cnt.bounding_rect for cnt in relevant]
        bounding_rects = sorted(bounding_rects, key=lambda x: x[0])  # Sort by
        # x value

        gaps = []
        for rect1, rect2 in zip(bounding_rects[:-1], bounding_rects[1:]):
            x1, _, w1, _ = rect1
            x2 = rect2[0]
            gaps.append(x2 - (x1 + w1))
        if not gaps:  # Only one word in the line
            return [(0, self.__img.shape[1])]

        space_threshold = self.__get_space_threshold(gaps)
        spaces = []
        for rect1, rect2 in zip(bounding_rects[:-1], bounding_rects[1:]):
            x1, _, w1, _ = rect1
            x2 = rect2[0]
            gap = x2 - (x1 + w1)
            if gap >= space_threshold:
                spaces.append((x2 + x1 + w1) / 2)

        spaces = [0] + spaces + [self.__img.shape[1]]
        return zip(spaces[:-1], spaces[1:])

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
        rescaled = np.array(pil_img, dtype=np.float64)
        rescaled = rescaled[:, :, 0]  # BGR to grayscaled

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
        contours = [cnt.contour for cnt in contours]
        cv2.drawContours(mask, contours, -1, 255, -1)
        # Now crop
        x, y = np.where(mask == 255)
        top_x, top_y = (np.min(x), np.min(y))
        bottom_x, bottom_y = (np.max(x), np.max(y))
        return mask[top_x:bottom_x+1, top_y:bottom_y+1]

    def __get_cnt_directly_underneath(self, cnt):
        """
        Returns the contour directly beneath the one inputted.
        :param cnt: The contour in question.
        :return: The contour beneath it.
        """
        x, y, w, h = cnt.bounding_rect
        center_x = x + w / 2
        center_y = y + h / 2
        external = self.__get_external_contours()
        sorted_cnt = sorted(external, key=lambda _x: _x.bounding_rect[1])
        for contour in sorted_cnt:
            x2, y2, w2, h2 = contour.bounding_rect
            if y2 > center_y:
                if x2 < center_x < (x2 + w2):
                    return contour

    def __segment_tittles(self):
        """
        Tittles are the little dots above 'i' and 'j'. This function adds them
        to the letter contour.
        """
        external = self.__get_external_contours()
        if not external:
            return
        y_distances = []
        for cnt in external:
            cnt2 = self.__get_cnt_directly_underneath(cnt)
            if cnt2:
                x1, y1, w1, h1 = cnt.bounding_rect
                x2, y2, w2, h2 = cnt2.bounding_rect
                y_distances.append(y2 - (y1 + h1))
        if not y_distances:
            return
        average_y_distance = np.mean(y_distances)

        sorted_cnt = sorted(external, key=lambda x: x.bounding_rect[1])
        for cnt in sorted_cnt:
            cnt2 = self.__get_cnt_directly_underneath(cnt)
            if cnt2:
                x1, y1, w1, h1 = cnt.bounding_rect
                x2, y2, w2, h2 = cnt2.bounding_rect
                y_distance = y2 - (y1 + h1)
                if y_distance < average_y_distance * Y_DISTANCE_TO_AVE_RATIO:
                    cnt.hierarchy[3] = cnt2.index

    @staticmethod
    def __get_space_threshold(char_gaps):
        """
        Returns the minimum space threshold in a line by it's the gaps between
        the characters.
        :param char_gaps: [gap between character]
        :return: The minimum threshold for space.
        """
        char_gaps = sorted([max(0, gap) for gap in char_gaps])
        weights = range(1, len(char_gaps) + 1)  # + len(char_gaps) / 10
        char_average = np.average(char_gaps, weights=weights)
        last_gaps = char_gaps[-(len(char_gaps) / 4):]
        last_average = np.mean(last_gaps)
        return np.average([char_average, last_average], weights=[2, 1])

    @staticmethod
    def __filter_contour(contour, bounds):
        """
        Returns whether the contour is a character inside the word or not.
        Checks that the contour is an external one, if it's inside the bounds
        of the word, and if it is tall enough to not be a tittle (the little
        dots above 'i' and 'j').
        :param contour: The contour in question
        :param bounds: (upper, lower, left, right).
        :return: True if the contour is a character in the word. False
        otherwise.
        """
        upper, lower, left, right = bounds
        x, y, w, h = contour.bounding_rect
        x_center = x + w / 2
        y_center = y + h / 2
        if upper < y_center < lower and left < x_center < right:
            return True
        return False

    def __get_all_internal_contours(self, external_contour):
        """
        Returns all the internal contours within a contour.
        :param external_contour: The contour in question.
        """
        index = external_contour.index
        internal_contours = [cnt for cnt in self.__contours if
                             cnt.hierarchy[3] == index]
        internal_contours += [external_contour]
        return internal_contours

    def __separate_word(self, bounds):
        """
        Separate each word to its characters. (Also turns character to network
        format of 0-1 instead of 0-255.)
        :param bounds: (upper, lower, left, right) coordinates of the word.
        :return: List of characters.
        :rtype : [OpenCV2 image]
        """
        external = self.__get_external_contours()
        relevant = [cnt for cnt in external if
                    self.__filter_contour(cnt, bounds)]
        word_chars = []
        for cnt in relevant:
            contours = self.__get_all_internal_contours(cnt)
            char = self.__crop_contour(contours)
            x = cnt.bounding_rect[0]
            word_chars.append((char / 255.0, x))
        word_chars = [w[0] for w in sorted(word_chars, key=lambda a: a[1])]
        return word_chars

    def __separate_line(self, upper, lower):
        """
        Separate each line to its characters.
        :return: List of lists of characters (OpenCV2 image).
        :rtype : [words] = [[(OpenCV2 image)]]
        """
        words = self.__get_word_coords(upper, lower)
        line = []
        for left, right in words:
            chars = self.__separate_word((upper, lower, left, right))
            word_chars = [self.__rescale_char(char) for char in chars]
            # for i, char in enumerate(word_chars):
            #     cv2.imshow(str(i), char)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
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
        self.__segment_tittles()
        lines = self.__get_line_coords()
        text = []
        for upper, lower in lines:
            line_chars = self.__separate_line(upper, lower)
            if line_chars:
                text.append(line_chars)
        return text


# _path = 'D:\\School\\Programming\\Cyber\\FinalExercise-12th\\MNIST_Project\\' \
#         'code\\testing images\\whatsapp.jpeg'
#
# _img = cv2.imread(_path, 0)
# prep = Preprocessor(_img)
# _separated = prep.separate_text()
# for _line in _separated:
#     for _word in _line:
#         for _i, _char in enumerate(_word):
#             cv2.imshow(str(_i), _char)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()