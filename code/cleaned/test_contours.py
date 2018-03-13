# -*- coding: utf-8 -*-
import cv2
import numpy as np


def find_text_tilt(img):
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle


def rotate_to_text_tilt(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def get_line_coords(img, threshold):
    hist = cv2.reduce(img, 1, cv2.REDUCE_AVG).reshape(-1)
    h, w = img.shape[:2]
    uppers = [y for y in range(h - 1) if
              hist[y] <= threshold < hist[y + 1]]
    lowers = [y for y in range(h - 1) if
              hist[y] > threshold >= hist[y + 1]]
    return uppers, lowers


def get_word_coords(img, threshold):
    slices = cv2.reduce(img, 0, cv2.REDUCE_AVG).reshape(-1)
    count = 0
    is_line = False
    # y = 0
    line_coords = []
    for i, slc in enumerate(slices):
        if is_line:
            if not slc:
                is_line = False
                count = 1
                # y = i
        else:
            if slc:
                is_line = True
                if count >= threshold:
                    # line_coords.append(int(y / count))
                    line_coords.append(i - count / 2)
            else:
                # y += i
                count += 1
    return line_coords


def get_space_ratio(line_height):
    # return line_height / 2.67
    return line_height / 5


def main():
    im = cv2.imread('..\\testing text.png', 1)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(imgray, 215, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.bitwise_not(thresh)

    angle = find_text_tilt(thresh)
    rot_th = rotate_to_text_tilt(thresh, angle)

    rotated_gray = rotate_to_text_tilt(imgray, angle)
    rotated_gray = cv2.cvtColor(rotated_gray, cv2.COLOR_GRAY2BGR)

    uppers, lowers = get_line_coords(rot_th, 1)
    # rotated_thresh = cv2.cvtColor(rotated_thresh, cv2.COLOR_GRAY2BGR)

    for upper, lower in zip(uppers, lowers):
        space_len = get_space_ratio(lower - upper)
        words = get_word_coords(rot_th[upper:lower], space_len)
        # for x in words:
        #     cv2.line(rot_th, (x, upper), (x, lower), (255, 0, 0), 1)

        words.append(rot_th.shape[:2][1])

        for left, right in zip(words[:-1], words[1:]):
            # cv2.line(rotated_thresh, (left, upper), (right, upper),
            #          (255, 0, 0), 1)
            _, contours, hierarchy = cv2.findContours(rot_th[upper:lower,
                                                      left:right],
                                                      cv2.RETR_TREE,
                                                      cv2.CHAIN_APPROX_SIMPLE)
            contours = np.array(contours)
            contours = contours[np.where(hierarchy[0, :, 3] == -1)]
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if h < (lower - upper) / 3:
                    continue
                # print x, y, w, h
                # print hierarchy
                cv2.rectangle(rotated_gray[upper:lower, left:right], (x, y),
                              (x + w, y + h), (255, 0, 0), 1)
    cv2.imshow('bla', rotated_gray)
    cv2.imshow('bla2', rot_th)


    # cv2.imshow('im', im)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()