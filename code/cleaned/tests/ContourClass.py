# -*- coding: utf-8 -*-
import cv2
import numpy as np


class Contour():
    def __init__(self, cnt, index):
        self.contour = cnt
        self.index = index
        self.bounding_rect = cv2.boundingRect(cnt)

    def __crop_contour(self, img):
        # x, y, w, h, = self.bounding_rect
        # shape = (x + w + 10000, y + h + 10000)
        shape = img.shape

        mask = np.zeros(shape)
        # print self.contour
        cv2.drawContours(mask, [self.contour], -1, 255, -1)
        out = np.zeros(shape)
        # print np.where(mask == 255)
        out[mask == 255] = mask[mask == 255]
        # Now crop
        x, y = np.where(mask == 255)
        top_x, top_y = (np.min(x), np.min(y))
        bottom_x, bottom_y = (np.max(x), np.max(y))
        return out[top_x:bottom_x+1, top_y:bottom_y+1]

    def show_contour(self, img):
        cv2.imshow(str(self.index), self.__crop_contour(img))
        pass