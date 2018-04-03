# -*- coding: utf-8 -*-
import cv2
import numpy as np
from ContourClass import Contour


SPACE_RATIO = 2.5
LINE_THRESHOLD = 10
MIN_CNT_HEIGHT_LINE_RATIO = 1
MIN_CNT_SIZE_RATIO = 0.0000005

NET_INPUT_WIDTH = 28
NET_INPUT_HEIGHT = 28
NETWORK_INPUT_SIZE = (NET_INPUT_HEIGHT, NET_INPUT_WIDTH)
CHARACTER_SIZE = 20.0
BORDER_SIZE = 1, 1, 1, 1


class Preprocessor():
    def __init__(self, img):
        self.__img = self.__prepare_img(img)
        _, self.cnts, self.hierarchy = cv2.findContours(
            self.__img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = []
        for i, cnt in enumerate(self.cnts):
            self.contours.append(Contour(cnt, i))

    def get_image(self):
        return self.__img

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
        return thresh

    def __get_external_contours(self):
        externals = []
        indices = np.where(self.hierarchy[0, :, 3] == -1)
        for index in indices[0]:
            externals.append(self.contours[index])
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
                if y_distance < average_y_distance * 0.8:
                    self.hierarchy[0, cnt.index, 3] = cnt2.index

    def get_line_coords(self):
        external = self.__get_external_contours()
        # external = cv2.findContours(
        #     self.__img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        # cv2.imshow('', self.__img)
        # cv2.waitKey(0)
        # print len(external)

        # average_cnt_height = np.mean([cnt.bounding_rect[3] for cnt in
        #                               external])
        # sorted_contours = sorted(external, key=lambda x: x.bounding_rect[1])
        # for cnt in sorted_contours:
        #     cnt2 = self.__get_cnt_directly_underneath(cnt)
        #     if cnt2:
        #         x1, y1, w1, h1 = cnt.bounding_rect
        #         x2, y2, w2, h2 = cnt2.bounding_rect
        #         combined_size = y2 + h2 - y1
        #         if combined_size < average_cnt_height * 1.3:
        #             print (x1, y1, w1, h1), (x2, y2, w2, h2)

        line_breaks = []
        # sizes = [(cnt.bounding_rect[1], cnt.bounding_rect[3]) for cnt in
        #          external]
        # sizes = [(cv2.boundingRect(cnt)[1], cv2.boundingRect(cnt)[3]) for cnt in
        #          external]
        # sizes = sorted(sizes, key=lambda x: x[0])
        external = sorted(external, key=lambda x: x.bounding_rect[1])
        for cnt1, cnt2 in zip(external[:-1], external[1:]):
            x1, y1, w1, h1 = cnt1.bounding_rect
            x2, y2, w2, h2 = cnt2.bounding_rect
            overlap = (y1 + h1) - y2
            if overlap <= 0 or h2 / overlap > 3:
                # cnt1.show_contour(self.__img)
                # cnt2.show_contour(self.__img)
                # cv2.waitKey(0)
                line_break = (y2 + (y1 + h1)) / 2
                line_breaks.append(line_break)

        for line in line_breaks:
            cv2.line(self.__img, (0, line), (self.__img.shape[1], line),
                     (255,), 1)
        cv2.imshow('', cv2.resize(self.__img, (0, 0), fx=1, fy=1))
        cv2.waitKey(0)
        lines = [0] + line_breaks + [self.__img.shape[0]]
        return zip(lines[:-1], lines[1:])

    def __get_word_coords(self, upper, lower):
        pass


_path = '..\\..\\testing images\\text in book.jpg'
_img = cv2.imread(_path, 0)
prepr = Preprocessor(_img)

prepr.segment_tittles()
print prepr.get_line_coords()
cv2.waitKey(0)
# cv2.imshow('', prepr.get_image())
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print 'lines: ', len(_text)
# for _line in _text:
#     print '\twords: ', len(_line)
#     for _word in _line:
#         print '\t\tchars: ', len(_word)
#         for _i, _char in enumerate(_word):
#             cv2.imshow(str(_i), cv2.resize(_char * 255.0, (0, 0), fx=1, fy=1))
#             # cv2.imshow('thresh', cv2.resize(thresh, (0, 0), fx=.5, fy=.5))
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()