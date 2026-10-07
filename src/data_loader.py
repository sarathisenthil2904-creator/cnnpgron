import tensorflow as tf


def load_cifar10():
    """Load CIFAR-10 dataset from Keras and normalize pixel values."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values from 0-255 to 0-1 for faster, stable training
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Flatten labels to 1D arrays for sparse categorical crossentropy
    y_train = y_train.reshape(-1).astype("int32")
    y_test = y_test.reshape(-1).astype("int32")

    return x_train, y_train, x_test, y_test
