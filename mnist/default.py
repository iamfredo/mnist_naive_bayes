from pathlib import Path
import cv2
import numpy as np

import pandas as pd
from sklearn import naive_bayes
from sklearn.metrics import classification_report


def train_test_data(mod: str,
                    model_selection: list,
                    X_train, y_train, X_test, y_test):
    for method in model_selection:
        model_name = str(method.__name__).upper()
        print(model_name, "\n")
        nb = method()
        nb.fit(X_train, y_train)
        pred = nb.predict(X_test)
        print("Never predicted values: ", set(y_test) - set(pred), "\n")
        # print(classification_report(y_test, pred, zero_division=0))
        print(f"{model_name}" + str(classification_report(y_test, pred, zero_division=0)))

        path = Path(f"../data/data_final/{mod}.txt")
        with open(path, "a") as f:
            f.write(f"\n\n {mod}:\n {model_name}: " + str(classification_report(y_test, pred, zero_division=0)))

        print("\n\n")


#sort (Pixel als Menge betrachten)
def sort_data(test, train):
    for i in range(test.shape[0]):
        test[i] = np.sort(test[i])
    for j in range(train.shape[0]):
        train[j] = np.sort(train[j])
    return test, train


# NORM
def norm_data(test, train):
    test = (test != 0).astype(float)
    train = (train != 0).astype(float)
    return test, train


# ZOOM
def zoom_in(arr):
    my_arr = []
    for elem in arr:
        p = np.array(elem, dtype="uint8")
        # p = elem
        p = p.reshape((28,28))
        x,y,w,h = cv2.boundingRect(p)
        img1 = p[y:(y+h), x:(x+w)]
        imgResized = cv2.resize(img1, (20,20))
        my_arr.append(imgResized)
    my_arr = np.array(my_arr)
    return my_arr


def count(testdata, traindata):
    testdata = (testdata != 0).astype(float)
    traindata = (traindata != 0).astype(float)

    trainlist = []
    testlist = []

    for j in range(testdata.shape[0]):
        testlist.append(np.array([np.sum(testdata[j]), 0]))

    for j in range(traindata.shape[0]):
        trainlist.append(np.array([np.sum(traindata[j]), 0]))

    return np.array(testlist), np.array(trainlist)


# Load Data
train_df = pd.read_csv("../data/mnist_train.csv")
test_df = pd.read_csv("../data/mnist_test.csv")
X_train_2d = train_df.drop("label", axis=1).values.reshape(60000, 28, 28)
X_test_2d = test_df.drop("label", axis=1).values.reshape(10000, 28, 28)
X_train_1d = X_train_2d.reshape(60000, 784)
X_test_1d = X_test_2d.reshape(10000, 784)
y_train = train_df["label"]
y_test = test_df["label"]

methods = [naive_bayes.MultinomialNB, naive_bayes.GaussianNB, naive_bayes.BernoulliNB]






############ AUSWERTUNG #############

### Performance without manipulation
train_test_data("default", methods, X_train_1d[:10000], y_train[:10000], X_test_1d[:2000], y_test[:2000])
test, train = count(X_test_1d[:2000], X_train_1d[:10000])
train_test_data("default NAIVE", methods, train, y_train[:10000], test, y_test[:2000])



### Normalize
test_norm, train_norm = norm_data(X_test_1d[:2000], X_train_1d[:10000])
train_test_data("default NORMALIZE", methods, train_norm, y_train[:10000], test_norm, y_test[:2000])

### Sort
test_sort, train_sort = sort_data(X_test_1d[:2000], X_train_1d[:10000])
train_test_data("default SORT", methods, train_sort, y_train[:10000], test_sort, y_test[:2000])

"""
import matplotlib.pyplot as plt
### Zoom in
plt.imshow(X_train_2d[5], cmap="gray")
plt.show()
X_train_crop = zoom_in(X_train_2d[:10000]).reshape(10000, 400)
X_test_crop = zoom_in(X_test_2d[:2000]).reshape(2000, 400)
plt.imshow(X_train_crop[5].reshape((20,20)), cmap="gray")
plt.show()
# train_test_data("default ZOOM IN", methods, X_train_crop, y_train[:10000], X_test_crop, y_test[:2000])
"""

### Distribution of classes
y_count = np.zeros(10)
for label in y_train[:10000]:
    y_count[label] += 1

print("y_counts: ", y_count, y_count.sum(), "\n")

# Probabilities per Label
label_probs = y_count/y_count.sum()*100
for i in range(10):
    print(f"Probability for Class {i}: {label_probs[i]:.2f} %")
