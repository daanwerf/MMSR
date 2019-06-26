'''
#Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import numpy as np
import os

def load_data():
    """Loads COREL10k dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    # 8000 training samples, 2000 test samples
    x_train = np.empty((8000, 768, 1), dtype='uint8')
    y_train = np.empty((8000,), dtype='uint8')

    x_test = np.empty((2000, 768, 1), dtype='uint8')
    y_test = np.empty((2000,), dtype='uint8')

    count = 1
    train_count = 0
    test_count = 0
    for i in range(100):
        print(str(i) + "%")
        for j in range(100):
            filename = str(i) + "_" + str(count)
            p = np.load('featuredb/' + filename + ".npy")

            if j < 80:
                x_train[train_count] = p
                y_train[train_count] = i
                train_count = train_count + 1
            else:
                x_test[test_count] = p
                y_test[test_count] = i
                test_count = test_count + 1

            count = count + 1

    return (x_train, y_train), (x_test, y_test)

print(load_data())