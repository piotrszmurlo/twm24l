import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

from pathlib import Path
from imageio.v2 import imread
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from hog import compute_hog_features
from training_data import get_negative_images, get_positive_images


def train_svm() -> SVC:
    pos_images = []
    neg_images = []
    for i in range(5, 280):
        try:
            image = imread(Path(__file__).parents[2].joinpath('dataset', 'test', f'{str(i).zfill(3)}_set2.jpg'))
            image_greyscale = rgb2gray(image)
        except FileNotFoundError:
            continue
        boxes = []
        for line in open(Path(__file__).parents[2].joinpath('dataset', 'test', f'{str(i).zfill(3)}_set2.txt')):
            boxes.append([float(x) for x in line.strip().split(' ')])
        pos_images.extend(get_positive_images(image_greyscale, boxes, num_of_images=10))
        neg_images.extend(get_negative_images(image_greyscale, boxes, num_of_images=10))
        print(f'Processed image: {i}')
    pos_hogs = []
    for i, img in enumerate(pos_images):
        print(f'Calculating a HOG for a positive image {i}')
        pos_hogs.append(
            compute_hog_features(
                img,
                n_orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(1, 1)
            )
        )
    neg_hogs = []
    for i, img in enumerate(neg_images):
        print(f'Calculating a HOG for a negative image {i}')
        neg_hogs.append(
            compute_hog_features(
                img,
                n_orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(1, 1)
            )
        )
    pos_labels = np.ones(len(pos_hogs))
    neg_labels = np.zeros(len(neg_hogs))

    x = np.asarray(pos_hogs + neg_hogs)
    y = np.asarray(list(pos_labels) + list(neg_labels))
    print(f'x shape: {x.shape}')
    print(f'y shape: {y.shape}')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print('Starting SVM training')
    svc = SVC()
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    print(f'SVM accuracy: {accuracy_score(y_pred=y_pred, y_true=y_test) * 100}%')
    return svc


def sliding_window(
        image: np.ndarray,
        window_size: tuple[int, int] = (200, 200),
        step: int = 50
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    coords = []
    features = []
    img_height, img_width = image.shape[:2]
    for w1, w2 in zip(range(0, img_width - window_size[0], step), range(window_size[0], img_width, step)):
        for h1, h2 in zip(range(0, img_height - window_size[1], step), range(window_size[1], img_height, step)):
            print(f'Processing sliding window with coords: {w1}, {w2}, {h1}, {h2}')
            window = image[h1:h2, w1:w2]
            features_of_window = compute_hog_features(
                window,
                n_orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(1, 1),
            )
            coords.append((w1, w2, h1, h2))
            features.append(features_of_window)
    return (coords, np.asarray(features))


def change_heatmap_region_value(heatmap: np.ndarray, coords: tuple[int, int, int, int], value: int) -> np.ndarray:
    w1, w2, h1, h2 = coords
    heatmap[h1:h2, w1:w2] = heatmap[h1:h2, w1:w2] + value
    return heatmap


def process_heatmap(heatmap: np.ndarray, threshold: int = 170) -> np.ndarray:
    scaler = MinMaxScaler()
    heatmap = scaler.fit_transform(heatmap)
    heatmap = np.asarray(heatmap * 255).astype(np.uint8)
    heatmap = cv2.inRange(heatmap, threshold, 255)
    return heatmap


def detect(image: np.ndarray, svm: SVC):
    image_greyscale = rgb2gray(image)
    coords, features = sliding_window(image_greyscale)
    heatmap = np.zeros(image_greyscale.shape[:2])

    for i in range(len(features)):
        decision = svm.predict([features[i]])
        if decision[0] == 1:
            heatmap = change_heatmap_region_value(heatmap, coords[i], 30)
        else:
            heatmap = change_heatmap_region_value(heatmap, coords[i], -30)
    heatmap = process_heatmap(heatmap)
    contours, _ = cv2.findContours(heatmap, 1, 2)[:2]
    for c in contours:
        if cv2.contourArea(c) < 50 * 50:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        image = cv2.rectangle(image, (x, y), (x+w, y+h), (255), 2)
    return image


if __name__ == '__main__':
    try:
        svm = pickle.load(open('svm.pickle', 'rb'))
        print('Using pre-trained svm from disk')
    except FileNotFoundError:
        svm = train_svm()
        pickle.dump(svm, open('svm.pickle', 'wb'))

    for i in range(5):
        try:
            image = imread(Path(__file__).parents[2].joinpath('dataset', 'test', f'{str(i).zfill(3)}_set2.jpg'))
        except FileNotFoundError:
            continue
        img_with_boxes = detect(image, svm)
        cv2.imwrite(f'hog_example_{i}.jpg', img_with_boxes)
        plt.imshow(img_with_boxes, interpolation='nearest')
        plt.show()
