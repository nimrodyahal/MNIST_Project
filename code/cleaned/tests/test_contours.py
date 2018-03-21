# -*- coding: utf-8 -*-
import cv2
import numpy as np
import Image


SPACE_RATIO = 3


def load_image(path):
    im = cv2.imread(path, 1)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(imgray, 215, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.threshold(imgray, 100, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.bitwise_not(thresh)

    angle = find_text_tilt(thresh)
    rot_th = rotate_to_text_tilt(thresh, angle)

    rot_gray = rotate_to_text_tilt(imgray, angle)
    rot_gray = cv2.cvtColor(rot_gray, cv2.COLOR_GRAY2BGR)
    return rot_th, rot_gray


def find_text_tilt(img):
    coords = np.column_stack(np.where(img > 0))
    angle = abs(cv2.minAreaRect(coords)[-1])
    if angle > 45:
        angle -= 90
    return angle


def rotate_to_text_tilt(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def get_line_coords(img, threshold):
    slices = cv2.reduce(img, 1, cv2.REDUCE_AVG).reshape(-1)
    count = 0
    is_line = False
    word_coords = []
    for i, slc in enumerate(slices):
        if is_line:
            if not slc:
                is_line = False
                count = 1
        else:
            if slc:
                is_line = True
                if count >= threshold:
                    word_coords.append(i - count / 2)
            else:
                count += 1
    word_coords.append(img.shape[:2][1])
    return zip(word_coords[:-1], word_coords[1:])


def get_line_height(line):
    contours = cv2.findContours(line, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[1]
    heights = [cv2.boundingRect(cnt)[3] for cnt in contours]
    return np.mean(heights)


def get_word_coords(img, threshold):
    slices = cv2.reduce(img, 0, cv2.REDUCE_AVG).reshape(-1)
    count = 0
    is_line = False
    word_coords = []
    for i, slc in enumerate(slices):
        if is_line:
            if not slc:
                is_line = False
                count = 1
        else:
            if slc:
                is_line = True
                if count >= threshold:
                    word_coords.append(i - count / 2)
            else:
                count += 1
    word_coords.append(img.shape[:2][1])
    return zip(word_coords[:-1], word_coords[1:])


def detect_contours(thresh_img, line_height):
    chars = []
    _, contours, hierarchy = cv2.findContours(thresh_img,
                                              cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours)
    for i, cnt in enumerate(contours):
        if i not in np.where(hierarchy[0, :, 3] == -1)[0]:
            continue
        h = cv2.boundingRect(cnt)[3]
        if h + 2 >= line_height / 5:
            hie = list(np.where(hierarchy[0, :, 3] == i)[0]) + [i]
            char = crop_contour(contours[hie], thresh_img.shape)
            chars.append(char)
            # cv2.imshow(str(i), char)
    return chars


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


def rescale_char(char):
    bs_t, bs_b, bs_l, bs_r = BORDER_SIZE
    char = cv2.copyMakeBorder(char, bs_t, bs_b, bs_l, bs_r,
                              cv2.BORDER_CONSTANT, value=[0, 0, 0])

    char = char.astype(np.uint8)
    cv2_im = cv2.cvtColor(char, cv2.COLOR_GRAY2BGR)
    pil_img = Image.fromarray(cv2_im)

    w, h = pil_img.size
    scaling = CHARACTER_SIZE / max(w, h)
    new_size = (int(scaling * w), int(scaling * h))

    pil_img = pil_img.resize(new_size, Image.ANTIALIAS)
    rescaled = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)

    rescaled = cv2.copyMakeBorder(rescaled, 4, 4, 8, 8,
                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return rescaled


# def tittle_segmentation(contours):
#     """
#     Tittle is the dot above 'i' and 'j'. It causes problems when segmenting
#     into individual characters, so this functions fixes that by comparing the
#     coordinates of different contour bounding boxes to find ones that are right
#     above one another, then combining them.
#     """
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         x2 = x + w
#         y2 = y + h
#     pass


CHARACTER_SIZE = 20.0
BORDER_SIZE = 1, 1, 1, 1


def main():
    thresh, gray = load_image('..\\..\\testing images\\testing text.png')

    lines = get_line_coords(thresh, 5)

    for upper, lower in lines:
        line = thresh[upper:lower]
        line_height = get_line_height(line)
        space_len = line_height / SPACE_RATIO
        words = get_word_coords(line, space_len)

        for left, right in words:
            chars = detect_contours(line[:, left:right], line_height)
            for char in chars:
                rescaled = rescale_char(char)
                cv2.imshow('resized', rescaled)
                cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

    # im = cv2.imread('..\\..\\testing images\\testing text.png', 1)
    # imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    # detect_contours(thresh, 100)
    # cv2.imshow('bla', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
