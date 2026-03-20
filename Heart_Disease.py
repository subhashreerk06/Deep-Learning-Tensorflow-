
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # or "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional

import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from keras.utils import plot_model


keras.utils.set_random_seed(42)

df = pd.read_csv("http://storage.googleapis.com/download.tensorflow.org/data/heart.csv")

print(print(f"Shape of original version = {df.shape}"))
print(df.target.value_counts(normalize=True, dropna=False))

categorical_variables = ['sex', 'cp', 'fbs', 'restecg','exang', 'ca', 'thal']
numerics = ['age', 'trestbps','chol', 'thalach', 'oldpeak', 'slope']

df = pd.get_dummies(df, columns = categorical_variables) #To create one-hot encoded version of the data 

print(f"Shape of one-hot encoded version = {df.shape}")

test_df = df.sample(frac=0.2, random_state=42) #To split 20% of the data into the test set 
print(f"Test set shape = {test_df.shape}")

train_df = df.drop(test_df.index)
print(f"Training set shape = {train_df.shape}")

means = train_df[numerics].mean()
sd = train_df[numerics].std()

print(f"Mean of the training set = {means}")
print(f"Standard deviation of the training set = {sd}")

train_df[numerics] = (train_df[numerics] - means)/sd  #To standardize the training set
test_df[numerics] = (test_df[numerics] - means)/sd #To standardize the test set 

#To convert into numpy arrays to feed into Keras 
train = train_df.to_numpy(dtype=np.float32)
test = test_df.to_numpy(dtype=np.float32)

#To remove the target from the training and the test sets 
train_X = np.delete(train, 6, axis=1)
test_X = np.delete(test, 6, axis=1)

train_y = train[:, 6]
test_y = test[:, 6]

num_columns = train_X.shape[1]

# define the input layer 

input = keras.Input(shape=(num_columns,))

#feed the input vector to the hidden layer 

h = keras.layers.Dense(16, activation="relu", name="Hidden")(input)

#feed the output of the hidden layer to the output layer
 
output = keras.layers.Dense(1, activation="sigmoid", name="Output")(h)

#tell Keras that this (input,output) pair is your model
model = keras.Model(input, output)

print(model.summary())


plot_model(model, to_file="model.png", show_shapes=True, dpi=200)
print("Saved model.png")

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])


history = model.fit(train_X,              # the array with the input X columns
                    train_y,              # the array with the output y column
                    epochs=300,           # number of epochs to run
                    batch_size=32,        # number of samples (ie data points) per batch
                    verbose=1,            # verbosity during training
                    validation_split=0.2) # use 20% of the data for validation


history_dict = history.history
history_dict.keys()

loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


