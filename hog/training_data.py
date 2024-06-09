import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

from pathlib import Path
from imageio.v2 import imread
from skimage.color import rgb2gray


def get_positive_images(
        img: np.ndarray,
        existing_boxes: list[list[int]],
        box_size: tuple[int, int] = (200, 200),
        num_of_images: int = 5,
) -> np.ndarray:
    pos_images = []
    for rand_box in random.choices(existing_boxes, k=num_of_images):
        box_start = (
            int((rand_box[1] - rand_box[3] / 2) * img.shape[1]),
            int((rand_box[2] - rand_box[4] / 2) * img.shape[0]),
        )
        square_size = int(max(rand_box[3] * img.shape[1], rand_box[4] * img.shape[0]))
        pos_img = img[
            box_start[1]:box_start[1] + square_size,
            box_start[0]:box_start[0] + square_size,
        ]
        output = np.zeros((square_size, square_size))
        output[:pos_img.shape[0], :pos_img.shape[1]] = pos_img
        output = cv2.resize(output, box_size, interpolation=cv2.INTER_CUBIC)
        output = cv2.resize(output, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        pos_images.append(output)
    return pos_images


def get_negative_images(
        img: np.ndarray,
        existing_boxes: list[list[float]],
        box_size: tuple[int, int] = (200, 200),
        num_of_images: int = 5,
) -> np.ndarray:
    size_x, size_y = box_size
    neg_images = []
    iterations = 0
    while len(neg_images) < num_of_images:
        if iterations > num_of_images * 10:
            break
        box_start = (
            int(random.uniform(0, img.shape[1] - size_x)),
            int(random.uniform(0, img.shape[0] - size_y))
        )
        overlap = False
        for b in existing_boxes:
            output = np.zeros(box_size)
            if (
                not (
                    box_start[0] + size_x < (b[1] - b[3]/2) * img.shape[1]
                    or box_start[0] > (b[1] + b[3]/2) * img.shape[1]
                )
                or not (
                    box_start[1] + size_y > (b[2] - b[4]/2) * img.shape[0]
                    or box_start[1] < (b[2] + b[4]/2) * img.shape[0]
                )
            ):
                overlap = True
                break
        iterations += 1
        if overlap:
            continue
        neg_img = img[
            box_start[1]:box_start[1] + size_y,
            box_start[0]:box_start[0] + size_x,
        ]
        output[:neg_img.shape[0], :neg_img.shape[1]] = neg_img
        output = cv2.resize(output, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        neg_images.append(output)
    return neg_images


if __name__ == '__main__':
    image = imread(Path(__file__).parents[2].joinpath('dataset', 'test', '002_set2.jpg'))
    boxes = []
    image_greyscale = rgb2gray(image)
    for line in open(Path(__file__).parents[2].joinpath('dataset', 'test', '002_set2.txt')):
        boxes.append([float(x) for x in line.strip().split(' ')])
    pos_images = get_positive_images(image_greyscale, boxes, num_of_images=100)
    neg_images = get_negative_images(image_greyscale, boxes)
    for img in pos_images:
        plt.imshow(img, interpolation='nearest')
        plt.show()
    for img in neg_images:
        plt.imshow(img, interpolation='nearest')
        plt.show()
