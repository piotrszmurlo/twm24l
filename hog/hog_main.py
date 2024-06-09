import numpy as np
import cv2
import pickle

from pathlib import Path
from imageio.v2 import imread, imwrite
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

    train_image_paths = Path(__file__).parents[2].joinpath('dataset', 'train').glob('*.jpg')

    for img_path in train_image_paths:
        try:
            image = imread(img_path)
            # image = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
            image_greyscale = rgb2gray(image)
        except FileNotFoundError:
            continue
        boxes = []
        for line in open(str(img_path).replace('.jpg', '.txt')):
            boxes.append([float(x) for x in line.strip().split(' ')])
        pos_images.extend(get_positive_images(image_greyscale, boxes, box_size=(200, 200), num_of_images=3))
        neg_images.extend(get_negative_images(image_greyscale, boxes, box_size=(200, 200), num_of_images=12))
        print(f'Processed image: {img_path}')
    for idx, img in enumerate(pos_images):
        image_int = np.array(255*img, np.uint8)
        imwrite(Path(__file__).parents[2].joinpath('svm_train_dataset', 'pos', f'pos_img_{idx}.jpg'), image_int)
    for idx, img in enumerate(neg_images):
        image_int = np.array(255*img, np.uint8)
        imwrite(Path(__file__).parents[2].joinpath('svm_train_dataset', 'neg', f'neg_img_{idx}.jpg'), image_int)
    pos_hogs = []
    for i, img in enumerate(pos_images):
        print(f'Calculating a HOG for a positive image {i}')
        pos_hogs.append(
            compute_hog_features(
                img,
                n_orientations=9,
                pixels_per_cell=(4, 4),
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
                pixels_per_cell=(4, 4),
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
    svc = SVC(probability=True, class_weight={0: 4, 1: 1})
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    print(f'SVM accuracy: {accuracy_score(y_pred=y_pred, y_true=y_test) * 100}%')
    return svc


def sliding_window(
        image: np.ndarray,
        window_size: tuple[int, int] = (50, 50),
        step: int = 25,
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
                pixels_per_cell=(4, 4),
                cells_per_block=(1, 1),
            )
            coords.append((w1, w2, h1, h2))
            features.append(features_of_window)
    return (coords, np.asarray(features))


def change_heatmap_region_value(heatmap: np.ndarray, coords: tuple[int, int, int, int], value: int) -> np.ndarray:
    w1, w2, h1, h2 = coords
    heatmap[h1:h2, w1:w2] = heatmap[h1:h2, w1:w2] + value
    return heatmap


def process_heatmap(heatmap: np.ndarray, threshold: int = 200) -> np.ndarray:
    scaler = MinMaxScaler()
    heatmap = scaler.fit_transform(heatmap)
    heatmap = np.asarray(heatmap * 255).astype(np.uint8)
    # heatmap = cv2.inRange(heatmap, threshold, 255)
    return heatmap


def detect(image: np.ndarray, svm: SVC):
    image_greyscale = rgb2gray(image)
    coords, features = sliding_window(image_greyscale)
    heatmap = np.zeros(image_greyscale.shape[:2])

    for i in range(len(features)):
        probabilities = svm.predict_proba([features[i]])
        heatmap = change_heatmap_region_value(heatmap, coords[i], probabilities[0][1])
    heatmap = process_heatmap(heatmap)
    blur = cv2.GaussianBlur(heatmap, (13, 13), 11)
    heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, image, 0.5, 0)
    heatmap[heatmap < 128] = 0
    contours, _ = cv2.findContours(heatmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[:2]
    for c in contours:
        if cv2.contourArea(c) < 50 * 50:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        image = cv2.rectangle(image, (x, y), (x+w, y+h), (255), 2)
    return super_imposed_img, image, [c for c in contours if cv2.contourArea(c) >= 50 * 50]


if __name__ == '__main__':
    try:
        svm = pickle.load(open('svm.pickle', 'rb'))
        print('Using pre-trained svm from disk')
    except FileNotFoundError:
        svm = train_svm()
        pickle.dump(svm, open('svm.pickle', 'wb'))

    test_image_paths = Path(__file__).parents[2].joinpath('dataset', 'test').glob('*.jpg')
    true_pos_list = []
    false_pos_list = []
    hit_rate_list = []
    for i, img_path in enumerate(test_image_paths):
        try:
            image = imread(img_path)
            image = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        except FileNotFoundError:
            continue
        heatmap, img_with_boxes, contours = detect(image, svm)
        boxes = [[float(x) for x in line.strip().split(' ')] for line in open(str(img_path).replace('.jpg', '.txt'))]
        true_pos = 0
        false_pos = 0
        for c in contours:
            match_found = False
            (x, y, w, h) = cv2.boundingRect(c)
            for box in boxes:
                pixel_box_start = (
                    int((box[1] - box[3] / 2) * image.shape[1]),
                    int((box[2] - box[4] / 2) * image.shape[0]),
                )

                x_left = max(x, pixel_box_start[0])
                y_top = max(y, pixel_box_start[1])
                x_right = min(x + w, int(pixel_box_start[0] + box[3] * image.shape[1]))
                y_bottom = min(y + h, int(pixel_box_start[1] + box[4] * image.shape[0]))

                if x_right < x_left or y_bottom < y_top:
                    iou = 0.0
                    continue
                intersection_area = (x_right - x_left) * (y_bottom - y_top)

                bb1_area = w * h
                bb2_area = (box[3] * image.shape[0]) * (box[4] * image.shape[1])
                iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
                print(iou)
                if iou > 0.15:
                    true_pos += 1
                    match_found = True
                    break
            if not match_found:
                false_pos += 1
        true_pos_list.append(true_pos)
        false_pos_list.append(false_pos)
        hit_rate_list.append(true_pos/len(boxes))
        heatmap = cv2.resize(heatmap, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        img_with_boxes = cv2.resize(img_with_boxes, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f'final\\heatmaps\\heatmap_final_{i}.jpg', heatmap)
        cv2.imwrite(f'final\\boxes\\hog_final_{i}.jpg', img_with_boxes)
    number_of_images = 210
    print(f'True positives - max: {max(true_pos_list)}, min: {min(true_pos_list)}, avg: {sum(true_pos_list)/number_of_images}')
    print(f'Hit rate - max: {max(hit_rate_list)}, min: {min(hit_rate_list)}, avg: {sum(hit_rate_list)/number_of_images}')
    print(f'False positives - max: {max(false_pos_list)}, min: {min(false_pos_list)}, avg: {sum(false_pos_list)/number_of_images}')
    ppv_list = [true_pos/(true_pos + false_pos) for true_pos, false_pos in zip(true_pos_list, false_pos_list)]
    print(f'Average positive predictive value (PPV) - max: {max(ppv_list)}, min: {min(ppv_list)}, avg: {sum(ppv_list)/number_of_images}')
    fdr_list = [false_pos/(true_pos + false_pos) for true_pos, false_pos in zip(true_pos_list, false_pos_list)]
    print(f'Average false discovery rate (FDR) - max: {max(fdr_list)}, min: {min(fdr_list)}, avg: {sum(fdr_list)/number_of_images}')
