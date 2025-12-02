import os
import numpy as np
from PIL import Image

def load_dataset(dataset_dir):
    X = []
    y = []
    classes = sorted(os.listdir(dataset_dir))

    for label_idx, label in enumerate(classes):
        folder = os.path.join(dataset_dir, label)
        if not os.path.isdir(folder):
            continue
        
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            try:
                img = Image.open(path).convert("L").resize((28, 28))
                X.append(np.array(img))
                y.append(label_idx)
            except:
                continue

    return np.array(X), np.array(y), classes
