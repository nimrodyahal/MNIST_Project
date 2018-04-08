# -*- coding: utf-8 -*-
import cv2
import numpy as np
from ContourClass import Contour


Y_DISTANCE_TO_AVE_RATIO = 0.4
MIN_CNT_SIZE_RATIO = 0.0000005

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

    def __prepare_img(self, img):
        """
        Returns a rotated to text tilt threshold of the image. Uses
        IMG_IMG_THRESHOLD to determine the threshold value.
        :param img: The image in question (OpenCV2 image).
        :return: A rotated to text tilt threshold of the image (OpenCV2 image).
        """
        blur = cv2.bilateralFilter(img, 40, 10, 10)
        thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 201, 30)
        thresh = cv2.bitwise_not(thresh)
        thresh = self.__clean_image(thresh)
        # cv2.imshow('thresh', thresh)

        # angle = self.__find_text_tilt(thresh)
        # rotated = self.__rotate_image(thresh, angle)
        # cv2.imshow('rotated', cv2.resize(rotated, (0, 0), fx=.5, fy=.5))
        # cv2.waitKey(0)
        return thresh

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

    def __get_external_contours(self):
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

    def __get_cnt_directly_underneath(self, cnt):
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

    def segment_tittles(self):
        external = self.__get_external_contours()
        y_distances = []
        for cnt in external:
            cnt2 = self.__get_cnt_directly_underneath(cnt)
            if cnt2:
                x1, y1, w1, h1 = cnt.bounding_rect
                x2, y2, w2, h2 = cnt2.bounding_rect
                y_distances.append(y2 - (y1 + h1))
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

    def get_line_coords(self):
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

        for line in line_breaks:
            cv2.line(self.__img, (0, line), (self.__img.shape[1], line),
                     (255,), 1)
        cv2.imshow('', cv2.resize(self.__img, (0, 0), fx=1, fy=1))
        cv2.waitKey(0)
        lines = [0] + line_breaks + [self.__img.shape[0]]
        return zip(lines[:-1], lines[1:])

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

    def __crop_contour(self, contours):
        """
        Redraws the character so that only the relevant parts remain.
        :param contours: List of all the contours that make the character.
        :return: The redrawn character (OpenCV2 image).
        """
        mask = np.zeros(self.__img.shape)
        cv2.drawContours(mask, contours, -1, 255, -1)
        # Now crop
        x, y = np.where(mask == 255)
        top_x, top_y = (np.min(x), np.min(y))
        bottom_x, bottom_y = (np.max(x), np.max(y))
        return mask[top_x:bottom_x+1, top_y:bottom_y+1]

    @staticmethod
    def __get_space_threshold(char_gaps):
        char_gaps = sorted([max(0, gap) for gap in char_gaps])
        print char_gaps
        weights = np.array(range(len(char_gaps)))  # + len(char_gaps) / 10
        return np.average(char_gaps, weights=weights)

    def get_word_coords(self, upper, lower):
        bounds = upper, lower, 0, self.__img.shape[1]
        external = self.__get_external_contours()
        relevant = [cnt for cnt in external if
                    self.__filter_contour(cnt, bounds)]
        bounding_rects = [cnt.bounding_rect for cnt in relevant]
        bounding_rects = sorted(bounding_rects, key=lambda x: x[0])
        gaps = []
        for rect1, rect2 in zip(bounding_rects[:-1], bounding_rects[1:]):
            x1, _, w1, _ = rect1
            x2 = rect2[0]
            gaps.append(x2 - (x1 + w1))
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


_path = '..\\..\\testing images\\TEST3.jpg'
_img = cv2.imread(_path, 0)
prepr = Preprocessor(_img)

prepr.segment_tittles()
_lines = prepr.get_line_coords()
for line in _lines:
    for word in prepr.get_word_coords(line[0], line[1]):
        cv2.imshow(str(word[0]), _img[line[0]:line[1], word[0]:word[1]])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# cv2.imshow('', prepr.get_image())
# cv2.waitKey(0)
# cv2.destroyAllWindows()
