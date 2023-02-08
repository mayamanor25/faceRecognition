import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt


def one_hot_array(Y):
    b = np.zeros((Y.size, Y.max() + 1))
    b[np.arange(Y.size), Y] = 1
    return b.T


def plot_digit(images, digits, index):
    image = images[:, index]
    digit = digits[:, index].argmax()
    im_reshape = image.reshape(28, 28)
    plt.imshow(im_reshape, cmap='Greys')
    plt.title("The label is: " + str(digit))
    plt.show()


def load_data():
    data = pd.read_csv("database10lines.csv")

    X = train.iloc[:, 1:].values
    Y = train.iloc[:, 0]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    x_train = np.array()
    y_train = np.array()
    x_test = np.array()
    y_test = np.array()

    X_train = X_train.T
    X_test = X_test.T
    Y_train = one_hot_array(Y_train.values)
    Y_test = one_hot_array(Y_test.values)

    print(f"{datetime.now()} Shape of X_train is: " + str(X_train.shape))
    print(f"{datetime.now()} Shape of X_test is: " + str(X_test.shape))
    print(f"{datetime.now()} Shape of Y_train is: " + str(Y_train.shape))
    print(f"{datetime.now()} Shape of Y_test is: " + str(Y_test.shape))

    return X_train, Y_train, X_test, Y_test


"""
for השורה השמאלית in data:
    if השורה השמאלית == "train":
        x_train = השורה האמצעית
        y_train = השורה הימנית
    else:
        x_train = השורה האמצעית
        x_test = השורה הימנית
"""
