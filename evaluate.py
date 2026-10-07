import os
from pathlib import Path

import tensorflow as tf

from src.data_loader import load_cifar10


MODEL_PATH = Path("saved_models/cifar10_cnn.keras")


def main():
    """Load a saved model and evaluate it on the CIFAR-10 test set."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Please run train.py first."
        )

    print("Loading dataset...")
    _, _, x_test, y_test = load_cifar10()

    print("Loading saved model...")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    print("Evaluating model...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
