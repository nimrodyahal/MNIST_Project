# -*- coding: utf-8 -*-
import cv2


class Contour():
    def __init__(self, cnt, index, hierarchy):
        self.contour = cnt
        self.index = index
        self.bounding_rect = cv2.boundingRect(cnt)
        self.hierarchy = hierarchy
