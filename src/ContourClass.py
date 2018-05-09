# -*- coding: utf-8 -*-
import cv2


class Contour():
    """
    A helper class for the Preprocessor. Prevents the Preprocessor to calculate
    the info of a contour many times, and instead just stores all the needed
    information in a single class.
    """
    def __init__(self, cnt, index, hierarchy):
        self.contour = cnt
        self.index = index
        self.bounding_rect = cv2.boundingRect(cnt)
        self.hierarchy = hierarchy
