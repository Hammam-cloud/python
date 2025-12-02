import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical

DATASET_PATH = "dataset"  # dataset/A, dataset/B, ...

def load_dataset():
    images = []
    labels = []

    # read folders alphabetically: A, B, C, ...
    classes = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    print("Found classes:", classes)

    for idx, cls in enumerate(classes):
        class_folder = os.path.join(DATASET_PATH, cls)

        for filename in os.listdir(class_folder):
            path = os.path.join(class_folder, filename)

            try:
                img = Image.open(path).convert("L")
                img = img.resize((28, 28))
                img = np.array(img) / 255.0

                images.append(img)
                labels.append(idx)
            except Exception as e:
                print("Skipping file:", path, "Error:", e)

    X = np.array(images).reshape(-1, 28, 28, 1)
    y = to_categorical(labels, num_classes=len(classes))

    print(f"Loaded {len(X)} images.")

    return X, y, classes
