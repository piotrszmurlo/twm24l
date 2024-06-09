from imageio.v2 import imread
from pathlib import Path
from typing import NewType

import numpy as np

from skimage.color import rgb2gray


Height = NewType('Height', int)
Width = NewType('Width', int)
OrientationsNum = NewType('OrientationsNum', int)


def get_gradients(
        img_greyscale: np.ndarray[Height, Width]
) -> tuple[np.ndarray[Height, Width], np.ndarray[Height, Width]]:
    gradient_x = np.zeros(img_greyscale.shape)
    gradient_y = np.zeros(img_greyscale.shape)

    gradient_x[:, 1:-1] = (img_greyscale[:, 2:] - img_greyscale[:, :-2]) / 2
    gradient_y[1:-1, :] = (img_greyscale[2:, :] - img_greyscale[:-2, :]) / 2

    return gradient_x, gradient_y


def get_gradient_direction(
        gradient_x: np.ndarray[Height, Width], gradient_y: np.ndarray[Height, Width]
) -> np.ndarray[Height, Width]:
    return np.rad2deg(np.arctan2(gradient_y, gradient_x)) % 180


def get_gradient_magnitude(
        gradient_x: np.ndarray[Height, Width], gradient_y: np.ndarray[Height, Width]
) -> np.ndarray[Height, Width]:
    return np.sqrt(np.power(gradient_x, 2) + np.power(gradient_y, 2))


def compute_hog_cell(
        n_orientations: OrientationsNum, magnitudes: np.ndarray[Height, Width], orientations: np.ndarray[Height, Width]
) -> np.ndarray[OrientationsNum]:
    bin_width = int(180 / n_orientations)
    hog = np.zeros(n_orientations)
    for i in range(orientations.shape[0]):
        for j in range(orientations.shape[1]):
            orientation = orientations[i, j]
            lower_bin_idx = min(int(orientation / bin_width), n_orientations - 1)
            hog[lower_bin_idx] += magnitudes[i, j]

    return hog / (magnitudes.shape[0] * magnitudes.shape[1])


def normalize_vector(v: np.ndarray, eps: int = 1e-5):
    return v / np.sqrt(np.sum(v ** 2) + eps ** 2)


def compute_hog_features(
        image: np.ndarray,
        n_orientations: OrientationsNum = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (1, 1),
) -> np.ndarray:
    gradient_x, gradient_y = get_gradients(image)
    shape_y, shape_x = gradient_x.shape
    cell_x, cell_y = pixels_per_cell
    block_x, block_y = cells_per_block

    magnitudes = get_gradient_magnitude(gradient_x, gradient_y)
    orientations = get_gradient_direction(gradient_x, gradient_y)

    n_cells_x = int(shape_x / cell_x)
    n_cells_y = int(shape_y / cell_y)
    n_blocks_x = int(n_cells_x - block_x) + 1
    n_blocks_y = int(n_cells_y - block_y) + 1

    hog_cells = np.zeros((n_cells_y, n_cells_x, n_orientations))

    prev_x = 0
    for it_x in range(n_cells_x):
        prev_y = 0
        for it_y in range(n_cells_y):
            magnitudes_patch = magnitudes[prev_y:prev_y + cell_y, prev_x:prev_x + cell_x]
            orientations_patch = orientations[prev_y:prev_y + cell_y, prev_x:prev_x + cell_x]
            hog_cells[it_y, it_x] = compute_hog_cell(n_orientations, magnitudes_patch, orientations_patch)
            prev_y += cell_y
        prev_x += cell_x

    hog_blocks_normalized = np.zeros((n_blocks_y, n_blocks_x, n_orientations))

    for it_blocks_x in range(n_blocks_x):
        for it_block_y in range(n_blocks_y):
            hog_block = hog_cells[it_block_y:it_block_y + block_y, it_blocks_x:it_blocks_x + block_x].ravel()
            hog_blocks_normalized[it_block_y, it_blocks_x] = normalize_vector(hog_block)

    return hog_blocks_normalized.ravel()


if __name__ == '__main__':
    image = imread(Path(__file__).parents[2].joinpath('dataset', 'test', '000_set2.jpg'))
    image_greyscale = rgb2gray(image)

    hog_features = compute_hog_features(
        image_greyscale,
        n_orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(1, 1)
    )
    print(hog_features)
