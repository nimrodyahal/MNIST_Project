# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys


def rotate_to_text_tilt(img):
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # print angle
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def get_line_coords(img):
    slices = cv2.reduce(img, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32S)
    print slices
    # cv2.imshow('image', slices)
    count = 0
    is_line = False
    y = 0
    line_coords = []
    for i, _ in enumerate(slices):
        if is_line:
            if not slices[i]:
                is_line = False
                count = 1
                y = i
        else:
            if slices[i]:
                is_line = True
                line_coords.append(int(y / count))
            else:
                y += i
                count += 1
    return line_coords


def main():
    img_name = 'milstein-backing.jpg'
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY +
                              cv2.THRESH_OTSU)
    rotated = rotate_to_text_tilt(thresh)

    line_coords = get_line_coords(rotated)
    lines = []
    for line1, line2 in zip(line_coords[:-1], line_coords[1:]):
        lines.append(rotated[line1:line2, ])

    contours, hierarchy = cv2.findContours(lines[0], cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)[1:]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 10:
            cv2.rectangle(lines[0], (x, y), (x + w, y + h), (255, 0, 255), 1)
            cv2.imshow('asd', rotated)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()