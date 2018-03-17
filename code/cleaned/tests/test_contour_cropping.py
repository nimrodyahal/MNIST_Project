# -*- coding: utf-8 -*-
import numpy as np
import cv2


def crop_contour(contours, shape):
    mask = np.zeros(shape)
    cv2.drawContours(mask, contours, -1, 255, -1)
    out = np.zeros(shape)
    out[mask == 255] = mask[mask == 255]

    # Now crop
    x, y = np.where(mask == 255)
    topx, topy = (np.min(x), np.min(y))
    bottomx, bottomy = (np.max(x), np.max(y))
    return out[topx:bottomx+1, topy:bottomy+1]


def main():
    path = '..\\..\\testing images\\testing contour.png'
    im = cv2.imread(path, 1)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(imgray, 215, 255, cv2.THRESH_BINARY)[1]

    _, contours, hierarchy = cv2.findContours(255 - thresh, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours)
    for i, cnt in enumerate(contours):
        if i not in np.where(hierarchy[0, :, 3] == -1)[0]:
            continue
        hie = list(np.where(hierarchy[0, :, 3] == i)[0]) + [i]
        out = crop_contour(contours[hie], thresh.shape)
        cv2.imshow(str(i), out)

    # Show the output image
    # cv2.imshow('Output', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
