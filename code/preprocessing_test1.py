# -*- coding: utf-8 -*-
import cv2
import numpy as np
import imutils
import sys


def find_text_tilt(img):
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle


def rotate_to_text_tilt(img, angle):
    # coords = np.column_stack(np.where(img > 0))
    # angle = cv2.minAreaRect(coords)[-1]
    # if angle < -45:
    #     angle = -(90 + angle)
    # else:
    #     angle = -angle
    # print angle
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def get_line_coords(img, threshhold):
    slices = cv2.reduce(img, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32S)
    # print slices
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
                if count >= threshhold:
                    line_coords.append(int(y / count))
            else:
                y += i
                count += 1
    return line_coords


def get_word_coords(img):
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return get_line_coords(img, 3)


def draw_contours(img):
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY +
                           cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)[1]
    correct_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 5:
            correct_contours.append([y, y + h, x, x + w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
            cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 255), 1)
            # cv2.imshow('image1', img)
            # cv2.imshow('image1thresh', thresh)
            # cv2.waitKey(0)
    return correct_contours


def main():
    # img_name = 'milstein-backing.jpg'
    img_name = 'numbers.png'
    img = cv2.imread(img_name)
    gray = cv2.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY +
                           cv2.THRESH_OTSU)[1]
    angle = find_text_tilt(thresh)
    rotated_thresh = rotate_to_text_tilt(thresh, angle)
    rotated_gray = rotate_to_text_tilt(gray, angle)
    line_coords = get_line_coords(rotated_thresh, 3)
    line_coords.append(len(rotated_gray))
    contours = []
    for line1, line2 in zip(line_coords[:-1], line_coords[1:]):
        word_coords = get_word_coords(rotated_thresh[line1:line2])
        word_coords.append(len(rotated_gray[0]))
        for word1, word2 in zip(word_coords[:-1], word_coords[1:]):
            cnt = draw_contours(rotated_gray[line1:line2, word1:word2])
            # print 'y1:', cnt[0][0], line1
            cnt[0][0] += line1 #+ 2
            # print 'y2:', cnt[0][1], line1
            cnt[0][1] += line1 #+ 2
            # print 'x1:', cnt[0][2], word1
            cnt[0][2] += word1 #- 7
            # print 'x2:', cnt[0][3], word1
            cnt[0][3] += word1 #- 7
            contours += cnt
            # cv2.imshow('image2', rotated_gray)
            # cv2.waitKey(0)
    # print contours[0][0], contours[0][1], contours[0][2], contours[0][3]
    # cv2.imshow('test1', img[contours[0][0]:contours[0][1], contours[0][2]:contours[0][3]])
    # cv2.imshow('test2', img[contours[1][0]:contours[1][1], contours[1][2]:contours[1][3]])
    # cv2.imshow('test3', img[contours[2][0]:contours[2][1], contours[2][2]:contours[2][3]])
    # cv2.imshow('test4', img[contours[3][0]:contours[3][1], contours[3][2]:contours[3][3]])
    # cv2.imshow('test5', img[contours[4][0]:contours[4][1], contours[4][2]:contours[4][3]])
    # cv2.waitKey(0)
    return contours
            # cv2.line(rotated, (i, line1), (i, line2), (255, 0, 0))
    # print line_coords
    # for i in line_coords:
    #     cv2.line(rotated, (0, i), (len(rotated[0]), i), (255, 0, 0))
    # cv2.imshow('image2', rotated_gray)
    # cv2.waitKey(0)


if __name__ == '__main__':
    pass
    # main()