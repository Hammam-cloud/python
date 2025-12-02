# /src/predict.py
import os
import sys
import numpy as np
from PIL import Image
from src.model import load_trained_model
from src.preprocessing import preprocess_image_pil


def load_classes(model_dir):
    classes_file = os.path.join(model_dir, 'classes.txt')
    if not os.path.exists(classes_file):
        raise FileNotFoundError('classes.txt not found in model dir: ' + model_dir)
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def predict_image(model_path, image_pil):
    model = load_trained_model(model_path)
    model_dir = os.path.dirname(model_path)
    classes = load_classes(model_dir)

    arr = preprocess_image_pil(image_pil)  # shape (28, 28)
    arr = arr.reshape((1, 28, 28, 1))  # ensure correct shape for CNN

    preds = model.predict(arr)
    idx = int(np.argmax(preds, axis=1)[0])
    score = float(np.max(preds))
    return classes[idx], score


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python predict.py /path/to/model.h5 /path/to/image.png')
        sys.exit(1)

    model_path = sys.argv[1]
    img_path = sys.argv[2]

    pil = Image.open(img_path).convert("L")  # convert to grayscale
    cls, score = predict_image(model_path, pil)

    print('Predicted:', cls, 'score:', score)
