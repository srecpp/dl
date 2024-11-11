from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
print("Training data shape:", x_train.shape)
print("Validation data shape:", x_valid.shape)
plt.imshow(x_train[0], cmap='gray')
plt.title(f'Label: {y_train[0]}')
plt.show()
x_train = x_train.reshape(60000, 784)
x_valid = x_valid.reshape(10000, 784)
print("Reshaped training data shape:", x_train.shape)
x_train = x_train / 255.0
x_valid = x_valid / 255.0
num_categories = 10
y_train = keras.utils.to_categorical(y_train, num_categories)
y_valid = keras.utils.to_categorical(y_valid, num_categories)
print("First 10 categorical labels of training data:", y_train[0:10])
model = Sequential()
model.add(Dense(units=512, activation='relu', input_shape=(784,)))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, verbose=1, validation_data=(x_valid, y_valid))