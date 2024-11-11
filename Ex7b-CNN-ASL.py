import tensorflow.keras as keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, Dense

# Load datasets
train_df = pd.read_csv(r"sign_mnist_train.csv")
valid_df = pd.read_csv(r"sign_mnist_test.csv")

# Separate features and labels
y_train = train_df['label']
y_valid = valid_df['label']

# Remove the label column from features
del train_df['label']
del valid_df['label']

x_train = train_df.values
x_valid = valid_df.values

# Normalize the pixel values to be between 0 and 1
x_train = x_train / 255.0
x_valid = x_valid / 255.0

# Reshape the data to match the input shape for Conv2D (28x28x1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_valid = x_valid.reshape(-1, 28, 28, 1)

# Determine the number of classes
num_classes = y_train.max() + 1

# Adjust labels if they start from 1 instead of 0
if y_train.min() == 1:
    y_train -= 1
    y_valid -= 1

# One-hot encode the labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

# Print the shapes of the datasets
print(x_train.shape)
print(x_valid.shape)
print(y_train.shape)
print(y_valid.shape)

# Build the CNN model
model = Sequential()
model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))

model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))

model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))

model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=num_classes, activation="softmax"))

# Summary of the model architecture
model.summary()

# Compile the model
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_valid, y_valid))
