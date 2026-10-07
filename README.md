# CNN Image Classification with TensorFlow/Keras

This project trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset to classify images into 10 object categories.

## Features
- Loads and preprocesses the CIFAR-10 dataset
- Builds a CNN with Convolution, Pooling, Dropout, and Dense layers
- Trains and validates the model
- Evaluates the trained model on the test set
- Saves the trained model to disk
- Plots loss and accuracy curves
- Beginner-friendly, well-commented code

## Project structure

```text
cnn-image-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── train.py
├── evaluate.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   └── utils.py
├── saved_models/
│   └── .gitkeep
├── plots/
│   └── .gitkeep
└── .venv/
```

## Requirements
- Python 3.9+
- TensorFlow 2.15+
- Matplotlib
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sarathisenthil2904-creator/cnnpgron.git
   cd cnnpgron
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
This project uses the CIFAR-10 dataset from Keras.

- 60,000 color images of size 32x32
- 10 classes:
  - airplane
  - automobile
  - bird
  - cat
  - deer
  - dog
  - frog
  - horse
  - ship
  - truck

The dataset is automatically downloaded via `tensorflow.keras.datasets.cifar10` when you run the training script.

## Training
Run the training script:

```bash
python train.py
```

This script will:
- load the dataset
- split training data into train/validation sets
- build the CNN model
- train the model for a chosen number of epochs
- save the model to `saved_models/cifar10_cnn.keras`
- save training curves to `plots/training_history.png`

## Evaluation
To evaluate the saved model on the test set:

```bash
python evaluate.py
```

This prints the final test loss and accuracy and confirms the model works correctly on unseen data.

## Model output
- Saved model: `saved_models/cifar10_cnn.keras`
- Accuracy/loss plot: `plots/training_history.png`

## Usage tips
- If you want to adjust the architecture, edit `src/model.py`.
- If you want more training epochs, update `EPOCHS` in `train.py`.
- If you want to change the validation split, update the `split_validation_data` function in `src/data_loader.py`.

## Notes
- Training may take several minutes depending on your machine and hardware.
- GPU support is recommended but not required for running the project.
- If you use a CPU-only setup, training will still work, though slower.

## License
This project is intended for educational and learning purposes.
