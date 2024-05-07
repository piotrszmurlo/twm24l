import numpy as np
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
    size_x, size_y = box_size
    pos_images = []
    while len(pos_images) < num_of_images:
        output = np.zeros(box_size)
        rand_box = random.choice(existing_boxes)
        box_start = (
            int((rand_box[1] - rand_box[3] / 2) * img.shape[1]),
            int((rand_box[2] - rand_box[4] / 2) * img.shape[0]),
        )
        pos_img = img[
            box_start[1]:box_start[1] + size_y,
            box_start[0]:box_start[0] + size_x,
        ]
        output[:pos_img.shape[0], :pos_img.shape[1]] = pos_img
        pos_images.append(output)
    return pos_images


def get_negative_images(
        img: np.ndarray,
        existing_boxes: list[list[int]],
        box_size: tuple[int, int] = (200, 200),
        num_of_images: int = 5,
) -> np.ndarray:
    size_x, size_y = box_size
    neg_images = []
    while len(neg_images) < num_of_images:
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
        if overlap:
            continue
        neg_img = img[
            box_start[1]:box_start[1] + size_y,
            box_start[0]:box_start[0] + size_x,
        ]
        output[:neg_img.shape[0], :neg_img.shape[1]] = neg_img
        neg_images.append(output)
    return neg_images


if __name__ == '__main__':
    image = imread(Path(__file__).parents[2].joinpath('dataset', 'test', '002_set2.jpg'))
    boxes = []
    image_greyscale = rgb2gray(image)
    for line in open(Path(__file__).parents[2].joinpath('dataset', 'test', '002_set2.txt')):
        boxes.append([float(x) for x in line.strip().split(' ')])
    pos_images = get_positive_images(image_greyscale, boxes)
    neg_images = get_negative_images(image_greyscale, boxes)
    for img in pos_images:
        plt.imshow(img, interpolation='nearest')
        plt.show()
    for img in neg_images:
        plt.imshow(img, interpolation='nearest')
        plt.show()
