# src/predict.py
import os
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "model/alphabet_model.h5"

# Function to predict a single image array
def predict_image(model_path, img_array):
    """
    Predict the alphabet character from a 28x28 grayscale image array.

    Args:
        model_path (str): Path to the trained model (.h5)
        img_array (np.array): Shape (1, 28, 28, 1), normalized [0,1]

    Returns:
        str: Predicted character ('A'-'Z')
    """
    model = load_model(model_path)
    pred = model.predict(img_array)
    classes = [chr(i) for i in range(65, 91)]  # A-Z
    return classes[np.argmax(pred)]


# Optional: you can keep this main() for retraining
if __name__ == "__main__":
    from src.data_loader import load_dataset
    from src.model import build_model

    print("Loading dataset...")
    X, y, classes = load_dataset()

    print("Building model...")
    model = build_model(num_classes=len(classes))

    print("Training...")
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    print("Saving model...")
    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)
    print("Model saved at:", MODEL_PATH)
