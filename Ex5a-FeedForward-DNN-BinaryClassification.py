
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('diabetes.csv')


train, test = train_test_split(dataset, test_size=0.25, random_state=0, stratify=dataset['Outcome'])


train_X = train[train.columns[:8]]
test_X = test[test.columns[:8]]
train_Y = train['Outcome']
test_Y = test['Outcome']


model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))  # First hidden layer with 12 nodes
model.add(Dense(8, activation='relu'))  # Second hidden layer with 8 nodes
model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 node (sigmoid for binary classification)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(train_X, train_Y, epochs=150, batch_size=10)

_, accuracy = model.evaluate(test_X, test_Y)

print('Accuracy: %.2f' % (accuracy * 100))
