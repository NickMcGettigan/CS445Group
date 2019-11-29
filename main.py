#!/usr/bin/python

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History 
from sklearn.model_selection import train_test_split
import pandas as pd

# Load Data
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize Data
train_images = train_images / 255.0
test_images = test_images / 255.0

train_set, valid_set, train_labels_set, valid_labels_set = train_test_split(train_images, train_labels, test_size=0.1, shuffle= True)

# Setup architecture
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(32, activation='sigmoid'),
    keras.layers.Dense(32, activation='sigmoid'),
    keras.layers.Dense(10, activation='softmax')
])

# Setup hyperparameters of SGD.
sgd = optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False)
model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model and output history
history = History()
history = model.fit(train_set, train_labels_set, 
            epochs=5, 
            verbose=1, 
            validation_data=(valid_set, valid_labels_set),
            callbacks=[history])

# Test model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
print ("-------------------------")

# Save loss values
hist_df = pd.DataFrame(history.history)
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)







