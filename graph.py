import matplotlib.pyplot as plt

epochs = list(range(1, 11))
train_accuracy = [0.65, 0.72, 0.78, 0.83, 0.87, 0.88, 0.89, 0.91, 0.92, 0.93]
val_accuracy = [0.60, 0.68, 0.75, 0.80, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89]
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy over Epochs')
plt.legend()
plt.show()
