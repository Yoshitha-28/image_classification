# Image Classification with TensorFlow & Keras

This project demonstrates a simple image classification task using TensorFlow and Keras. The goal is to classify images of animals into three categories: **Cat**, **Dog**, and **Other**.

## Project Overview

The project includes the following steps:

1. **Mount Google Drive**: Access the dataset stored on Google Drive.
2. **Dataset Splitting**: Split the dataset into training (80%) and testing (20%) sets.
3. **Data Preprocessing**: Use `ImageDataGenerator` for data augmentation and normalization.
4. **CNN Model Building**: Build a Convolutional Neural Network (CNN) model for multi-class classification.
5. **Model Training**: Train the model on the training data and validate it on the test data.
6. **Model Saving**: Save the trained model to Google Drive for later use.
7. **Plotting Results**: Plot the training and validation accuracy and loss.
8. **Predictions on New Images**: Load the model and make predictions on new images.

## Requirements

- Python 3.x
- TensorFlow (Keras included)
- Google Colab (recommended for running this project)
- Matplotlib
- NumPy

## Getting Started

1. **Mount Google Drive**: 
   - Ensure your Google Drive is mounted on Colab by running the provided code:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```

2. **Dataset**: 
   - Ensure the dataset is structured correctly on Google Drive in the folder `animals/` and includes the following subfolders:
     - `Cat/`
     - `Dog/`
     - `Other/`

3. **Run the Script**: 
   - Split the data, preprocess the images, build and train the model.
   - After training, the model will be saved as `cat_vs_others_model.h5` in your Google Drive.

4. **Model Prediction**:
   - Load the model and make predictions on new images by placing them in the `predict_images/` folder on Google Drive.

## Code Explanation

### 1. Dataset Preparation

- **Data Split**: The dataset is divided into two sets: training and testing (80%-20%).
- **Data Augmentation**: Augment the training data with rotations, shifts, zooms, and flips to help the model generalize better.

### 2. Model Architecture

- **Convolutional Layers**: A sequence of convolutional layers with increasing depth (32, 64, 128 filters).
- **MaxPooling**: Downsampling the feature maps to reduce dimensionality.
- **Dense Layer**: A fully connected layer with 512 units.
- **Dropout**: Added to prevent overfitting by randomly setting a fraction of input units to 0.
- **Softmax Output**: The final layer is a softmax activation for multi-class classification.

### 3. Model Training

- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Loss Function**: Categorical crossentropy for multi-class classification.
- **Metrics**: Accuracy is tracked during training.

### 4. Evaluation and Prediction

- **Plots**: The script generates plots showing training and validation accuracy and loss over epochs.
- **Predictions**: The model can be used to predict new images stored in the `predict_images/` folder.

## Screenshots of Output

### Training and Validation Accuracy

![Training and Validation Accuracy](path_to_your_screenshot/training_validation_accuracy.png)

### Training and Validation Loss

![Training and Validation Loss](path_to_your_screenshot/training_validation_loss.png)

### Image Prediction Example

![Prediction Example](path_to_your_screenshot/prediction_example.png)

## Model Files

- `cat_vs_others_model.h5`: The trained model file.
- Place your images in the `predict_images/` folder to make predictions.

## Notes

- Ensure that the Python environment is compatible with TensorFlow (Python 3.8 or 3.9 is recommended).
- For local execution, TensorFlow may not be fully compatible with Python 3.11; consider using a different Python version if you encounter issues.
