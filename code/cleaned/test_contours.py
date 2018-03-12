# -*- coding: utf-8 -*-
import cv2
import numpy as np


def main():
    im = cv2.imread('numbers.png', 1)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    h, w = im.shape[:2]
    img = np.ones((h, w, 3), np.uint8) * 255
    to_remove = []
    for i, cnt in enumerate(contours):
        if hierarchy[0, i, 3] != 0:
            to_remove.append(cnt)
    contours = [x for i, x in enumerate(contours) if x not in to_remove]
    cv2.drawContours(img, contours, -1, (0, 0, 0), 1)
    cv2.imshow('bla.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()