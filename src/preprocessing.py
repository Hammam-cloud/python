# /src/preprocessing.py
from PIL import Image
import numpy as np
import os
import sys

IMG_SIZE = (28, 28)


def load_dataset(dataset_dir):
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X = []
    y = []

    for c in classes:
        folder = os.path.join(dataset_dir, c)
        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            path = os.path.join(folder, fname)
            try:
                img = Image.open(path).convert('L')
                img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 255.0
                X.append(arr)
                y.append(class_to_idx[c])
            except Exception as e:
                print('Skipping', path, '->', e)

    X = np.array(X)
    y = np.array(y, dtype=np.int32)
    return X, y, classes


def preprocess_image_pil(pil_img):
    img = pil_img.convert('L').resize(IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def preprocess_image_path(path):
    img = Image.open(path)
    return preprocess_image_pil(img)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        X, y, classes = load_dataset(sys.argv[1])
        print('Loaded dataset:')
        print('X shape:', X.shape)
        print('y shape:', y.shape)
        print('Classes:', classes)
    else:
        print('Usage: python preprocessing.py /path/to/dataset')
