import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#To initialize the seeds of different random number generators so that every time the program is run, same set of numbers are created
keras.utils.set_random_seed(42)
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

print(f"Shape of the training set: {x_train.shape, y_train.shape}")
print(f"Shape of the test set: {x_test.shape, y_test.shape}")

print(f"First 10 rows : {y_train[:10]}")

labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
def plot_acc_curves(history):
  plt.clf()
  history_dict = history.history
  acc = history_dict["accuracy"]
  val_acc = history_dict["val_accuracy"]
  epochs = range(1, len(acc) + 1)
  plt.plot(epochs, acc, "bo", label="Training acc")
  plt.plot(epochs, val_acc, "b", label="Validation acc")
  plt.title("Training and validation accuracy")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.show()

x_train = x_train/ 255.0
x_test = x_test/ 255.0

#To change the dimension from 28x28 to 28x28x1
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)  

# input layer
input = keras.Input(shape=x_train.shape[1:])

#
# the first convolutional block
#
# convolutional layer
x = keras.layers.Conv2D(30,                    # Number of filters
                        kernel_size=(2, 2),    # The shape of each filter
                        activation="relu",     # RELU activation as usual
                        name="Conv_1")(input)
# pooling layer
x = keras.layers.MaxPool2D()(x)
# end of first convolutional block

#
# the second convolutional block
#
# convolutional layer
x = keras.layers.Conv2D(30,                    # Number of filters
                        kernel_size=(2, 2),    # The shape of each filter
                        activation="relu",     # RELU activation as usual
                        name="Conv_2")(x)
# pooling layer
x = keras.layers.MaxPool2D()(x)
# end of second convolutional block

# flatten layer
x = keras.layers.Flatten()(x)

# fully-connected (dense) ReLU layer
x = keras.layers.Dense(256, activation="relu")(x)

# output softmax layer
output = keras.layers.Dense(10, activation="softmax")(x)


model = keras.Model(input, output)

model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


history = model.fit(x_train,
                    y_train,
                    batch_size=64,
                    epochs=10,
                    validation_split=0.2)

model.summary()
history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)
print(history.history["loss"])
print(history.history["val_loss"])
score = model.evaluate(x_test, y_test)
print(score)