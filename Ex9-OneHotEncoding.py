import numpy as np
from numpy import array, argmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

samples = [
    'Photosynthesis is the process by which plants make their food',
    'The theory of relativity was proposed by Albert Einstein',
    'DNA carries genetic information in living organisms',
    'The periodic table organizes chemical elements based on their properties',
    'Gravity is the force that attracts objects toward the center of the Earth'
]

token_index = {}
counter = 0
for sample in samples:
    for considered_word in sample.split():
        if considered_word not in token_index:
            token_index[considered_word] = counter
            counter += 1

print("Token Index:", token_index)

data = list(token_index.keys())
values = array(data)

print("Values:", values)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

print("Integer Encoded:", integer_encoded)

onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print("OneHot Encoded:\n", onehot_encoded)

# Inverting the one-hot encoding to get the original word back
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])

print("Inverted:", inverted)
