import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from src.preprocessing import load_dataset
from src.model import build_model

DATASET = "dataset"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.txt")

def main():
    print("Loading dataset...")
    X, y, classes = load_dataset(DATASET)
    print(f"Loaded {len(X)} images.")

    X = X.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y = to_categorical(y, num_classes=len(classes))

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Building model...")
    model = build_model(num_classes=len(classes))


    print("Training model...")
    model.fit(X, y, epochs=5, batch_size=32)

    print("Saving model...")
    model.save(MODEL_PATH)

    with open(CLASSES_PATH, "w") as f:
        for c in classes:
            f.write(c + "\n")

    print("DONE! Model saved to:", MODEL_PATH)

if __name__ == "__main__":
    main()
