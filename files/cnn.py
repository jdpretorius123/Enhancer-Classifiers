"""
Convolutional Neural Network.

This module allows the user to train, validate,
and test a deep neural network on DNA sequence
data to predict whether a sequence is an
enhancer.

Functions
---------
"""

import process
from typing import Any
import numpy as np
from sklearn.metrics import roc_curve, auc  # type: ignore
import matplotlib.pyplot as plt
import tensorflow as tf  # type: ignore
import keras  # type: ignore
from keras.models import Sequential  # type: ignore
from keras.layers import Input, Flatten, Conv1D, MaxPooling1D  # type: ignore
from keras.layers import Dense, BatchNormalization, Dropout  # type: ignore
from keras.optimizers import Adam  # type: ignore
from keras.losses import BinaryCrossentropy  # type: ignore

keras.utils.set_random_seed(1)

fastafiles: list[str] = [
    "sim1/train.fasta",
    "sim1/validation.fasta",
    "sim1/test.fasta",
]
textfiles: list[str] = [
    "sim1/train.txt",
    "sim1/validation.txt",
    "sim1/test.txt",
]
process.prepAllData(fastafiles, textfiles)

train: tuple[np.ndarray, np.ndarray] = process.getDataSets(textfiles[0])
xtrain: np.ndarray = train[0]
xtrain = process.reshape(xtrain)
ytrain: np.ndarray = train[1]

val: tuple[np.ndarray, np.ndarray] = process.getDataSets(textfiles[1])
xval: np.ndarray = val[0]
xval = process.reshape(xval)
yval: np.ndarray = val[1]

test: tuple[np.ndarray, np.ndarray] = process.getDataSets(textfiles[2])
xtest: np.ndarray = test[0]
xtest = process.reshape(xtest)
ytest: np.ndarray = test[1]

epoch: int = 60  # 3000
batch: int = 300  # 500
lr: float = 1e-2
dr: float = 0.5
convlayers: int = 2
denselayers: int = 3  # 10
layerwidth: int = 100
numfilters: int = 5
kernelsize: int = 7

model: Any = Sequential()
model.add(Input(shape=(249, 4)))
for _ in range(convlayers):
    model.add(Conv1D(filters=numfilters, kernel_size=kernelsize, activation="relu"))
    model.add(MaxPooling1D())
    model.add(BatchNormalization())
    model.add(Dropout(rate=dr))

model.add(Flatten())
for _ in range(denselayers):
    model.add(Dense(units=layerwidth, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dr))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(
    loss=BinaryCrossentropy(),
    optimizer=Adam(learning_rate=lr),
    metrics=["accuracy"],
)
history: Any = model.fit(
    xtrain,
    ytrain,
    validation_data=(xval, yval),
    epochs=epoch,
    batch_size=batch,
    # verbose=0,
)

# Plot training & validation loss values
plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.savefig("figs/cnn-loss.png")
plt.close()

# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.savefig("figs/cnn-acc.png")
plt.close()

results: Any = model.evaluate(xtest, ytest, batch_size=batch)
print(results)
print(f"Loss = {results[0]}")
print(f"Accuracy = {results[1]}")

ypred: list[np.ndarray] = model.predict(xtest, batch_size=batch)
fpr, tpr, _ = roc_curve(ytest, ypred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=2,
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.savefig("figs/cnn-auc.png")
plt.close()
