#This code is for a simple image classification task using TensorFlow and Keras. It includes the following steps:
#1. Mounting Google Drive to access the dataset.
#2. Splitting the dataset into training and testing sets (80-20 split).
#3. Preprocessing the images using ImageDataGenerator.
#4. Building a Convolutional Neural Network (CNN) model for classification.
#5. Training the model on the training set and validating it on the testing set.
#6. Saving the trained model to Google Drive.
#7. Plotting the training and validation accuracy and loss.
#8. Making predictions on new images and displaying the results.

#The code work best in Google Colab, as it uses Google Drive for data storage and retrieval.
#It is important to ensure that the dataset is organized correctly in the specified directory structure for the code to work properly.

#However it can be run in any environment with TensorFlow and Keras installed, provided that the dataset is accessible and the paths are adjusted accordingly.
#The problem remains that python 3.11 does not support TensorFlow yet, so the code will not run in a local environment with python 3.11.

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import shutil
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define the path to the original dataset (where the images are located)
dataset_dir = '/content/drive/MyDrive/animals'

# Categories (folders in the dataset directory)
categories = ['Cat', 'Dog', 'Other']

# Destination directories for the split data
train_dir = '/content/drive/MyDrive/train'  # Path where training images will be stored
test_dir = '/content/drive/MyDrive/test'    # Path where testing images will be stored

# Define the test size (80-20 split)
test_size = 0.2  # 20% for testing, 80% for training

# Clean up previous train and test directories if they exist
for folder in [train_dir, test_dir]:
    if os.path.exists(folder):
        shutil.rmtree(folder)

# Create directories for training and testing
for category in categories:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

# Function to split the data into train and test
def split_data():
    for category in categories:
        category_path = os.path.join(dataset_dir, category)
        images = os.listdir(category_path)

        # Shuffle the list of images to ensure random splitting
        random.shuffle(images)

        # Split the images into train and test
        num_images = len(images)
        num_test_images = int(num_images * test_size)  # 20% for testing
        num_train_images = num_images - num_test_images  # Remaining 80% for training

        # Split the dataset into testing and training images
        test_images = images[:num_test_images]
        train_images = images[num_test_images:num_images]  # 80% for training

        # Copy images to the corresponding directories
        for image in test_images:
            src = os.path.join(category_path, image)
            dst = os.path.join(test_dir, category, image)
            shutil.copy(src, dst)

        for image in train_images:
            src = os.path.join(category_path, image)
            dst = os.path.join(train_dir, category, image)
            shutil.copy(src, dst)

    print(f"Dataset split into train and test sets (test size = {test_size * 100}%)")

# Call the function to split the dataset
split_data()

# Image dimensions (this should match the input shape of your model)
img_width, img_height = 150, 150  # You can adjust this size based on your needs

# Create ImageDataGenerators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize the image data to range [0, 1]
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for validation data

# Load the training and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),  # Resize images to the required size
    batch_size=32,
    class_mode='categorical'  # Use 'categorical' for multi-class classification
)

valid_generator = valid_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

# Build the neural network model for classification
model = models.Sequential()

# Add a convolutional base (CNN layers)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output and add dense layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout layer to prevent overfitting
model.add(layers.Dense(3, activation='softmax'))  # Output layer for multi-class classification

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Adam optimizer with a learning rate of 0.001
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']  # Accuracy as the evaluation metric
)

# Train the model using the data generators
history = model.fit(
    train_generator,
    epochs=10,  # Number of training epochs
    validation_data=valid_generator,
    validation_steps=50  # Number of validation steps (optional, adjust as needed)
)

# Optionally, save the model to Google Drive
model.save('/content/drive/MyDrive/cat_vs_others_model.h5')

# Print class indices
print("Class indices:", train_generator.class_indices)
# Should show: {'Cat': 0, 'Dog': 1, 'Other': 2}
# Print model summary
print(model.summary())

# Plot accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Image prediction
model_path = '/content/drive/MyDrive/cat_vs_others_model.h5'  # your saved model
image_folder = '/content/drive/MyDrive/predict_images'      # folder containing new images
img_size = (150, 150)  # must match your training size

# --- Load model ---
model = load_model(model_path)

class_names = ['Cat', 'Dog', 'Other']  # Make sure this matches your training order

for fname in os.listdir(image_folder):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(image_folder, fname)
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    pred_class = np.argmax(preds[0])
    label = class_names[pred_class]
    confidence = preds[0][pred_class]

    plt.imshow(img)
    plt.title(f"{fname}\nPrediction: {label} ({confidence:.2f})")
    plt.axis('off')
    plt.show()
