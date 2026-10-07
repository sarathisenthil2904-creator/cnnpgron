import os
from pathlib import Path

import matplotlib.pyplot as plt

from src.data_loader import load_cifar10
from src.model import build_cnn_model


MODEL_DIR = Path("saved_models")
PLOT_DIR = Path("plots")
MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


def plot_history(history):
    """Create and save accuracy/loss graphs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"], label="Training Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Training Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plot_path = PLOT_DIR / "training_history.png"
    plt.savefig(plot_path)
    print(f"Training plots saved to: {plot_path}")
    plt.show()


def main():
    """Train the CNN on CIFAR-10 and save the model."""
    print("Loading CIFAR-10 dataset...")
    x_train, y_train, x_test, y_test = load_cifar10()

    # Reserve a validation split from the training set
    val_size = int(0.1 * len(x_train))
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train_final = x_train[val_size:]
    y_train_final = y_train[val_size:]

    print("Building CNN model...")
    model = build_cnn_model(input_shape=(32, 32, 3), num_classes=10)

    print("Training model...")
    history = model.fit(
        x_train_final,
        y_train_final,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=64,
        verbose=1,
    )

    # Save the trained model
    model_path = MODEL_DIR / "cifar10_cnn.keras"
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Plot accuracy and loss curves
    plot_history(history)

    # Evaluate on test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
