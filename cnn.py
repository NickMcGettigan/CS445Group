#!/usr/bin/python

import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import pandas as pd
from tensorflow.keras.callbacks import History
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data for CNN 
# Remove if first layer is not CNN
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

# Onehot encode
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

# Setup Model
model = keras.Sequential([
   keras.layers.Conv2D(16, kernel_size=9,
      activation='sigmoid',
      input_shape=(28,28,1)),
   
   keras.layers.Conv2D(8, kernel_size=5,
      activation='sigmoid'),
   
   keras.layers.Flatten(input_shape=(28, 28)),
   #keras.layers.Dropout(.15),
   keras.layers.Dense(32, activation='sigmoid'),
   #keras.layers.Dense(32, activation='sigmoid'),
   # output layer
   keras.layers.Dense(10, activation='softmax')
])


# Setup hyperparameters of SGD.
sgd = optimizers.SGD(learning_rate=0.1, momentum=0.1, nesterov=False)
model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model and save history
history = History()
history = model.fit(X_train, y_train, 
            epochs=30, 
            batch_size = 32,
            verbose=1, 
            validation_data=(X_test, y_test),
            callbacks=[history])


# Test model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc, "Loss: ", test_loss)
print ("-------------------------")

# Save loss values
hist_df = pd.DataFrame(history.history)
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
