import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

image_directory = r'C:\Users\User\OneDrive\Desktop\Projects\BrainTumor_Classification_DL\datasets'

no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')
data = []
labels = []

INPUT_SIZE = 64

# Load and preprocess images
for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(os.path.join(image_directory, 'no', image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        data.append(image)
        labels.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(os.path.join(image_directory, 'yes', image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        data.append(image)
        labels.append(1)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Normalize pixel values to be between 0 and 1
data = data / 255.0

# Reshape data for CNN
data = data.reshape(-1, INPUT_SIZE, INPUT_SIZE, 1)  # add one channel for grayscale

# Convert labels to one-hot encoding
labels = to_categorical(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)

# Create CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # 2 output classes: tumor or no tumor
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)

# Save the CNN model
model.save('BrainTumor10EpochsCategorical.h5')
