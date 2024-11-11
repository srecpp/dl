import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load datasets
train_df = pd.read_csv(r"sign_mnist_train.csv")
valid_df = pd.read_csv(r"sign_mnist_test.csv")

# Separate features and labels
y_train = train_df['label']
y_valid = valid_df['label']

# Drop the label column from features
del train_df['label']
del valid_df['label']

x_train = train_df.values
x_valid = valid_df.values

# Visualize the first 20 images
plt.figure(figsize=(40, 40))
num_images = 20
for i in range(num_images):
    row = x_train[i]
    label = y_train[i]
    image = row.reshape(28, 28)
    plt.subplot(1, num_images, i + 1)
    plt.title(label, fontdict={'fontsize': 30})
    plt.axis('off')
    plt.imshow(image, cmap='gray')
plt.show()

# Normalize the pixel values to be between 0 and 1
x_train = x_train / 255.0
x_valid = x_valid / 255.0

# Print maximum and minimum values in the training labels
print(f"Maximum value in y_train: {y_train.max()}")
print(f"Minimum value in y_train: {y_train.min()}")

# Number of classes (sign language alphabets)
num_classes = y_train.max() + 1

# Adjust labels if they start from 1 instead of 0
if y_train.min() == 1:
    y_train -= 1
    y_valid -= 1

# Convert labels to one-hot encoding if necessary
if y_train.ndim == 1:
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)

# Build the neural network model
model = Sequential()
model.add(Dense(units=512, activation='relu', input_shape=(784,)))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

# Summarize the model architecture
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=20, verbose=1, validation_data=(x_valid, y_valid))
