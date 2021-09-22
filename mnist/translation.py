import math
from pathlib import Path
import cv2

import numpy as np
import pandas as pd
import scipy as sp
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


def shift(cpytest, scale=5, vertical=False):
    """
    Returns 2D images
    """
    n = scale  # Skalierungsvariable
    x = 2   # Richtung der Translation: 1 -> vertikal, 2 -> horizontal

    shift_array = [(n, 0), (-n, 0), (0, n), (0, -n)]
    x = np.random.randint(1, 4)
    shift_test = []
    if vertical is False:
        pass
    else:
        x=1
    for i in range(cpytest.shape[0]):
        
        shift_test.append(sp.ndimage.interpolation.shift(cpytest[i], [shift_array[x][0], shift_array[x][1]], cval=0,
                                                         mode='constant', order=0))
    return shift_test


def get_euclidean_distance(arr, sort=False):
    """
    Function that transforms colored pixels to Euclidean distances
    """
    euclid_array = []
    euclid_array = []
    x_mid = (X_train_2d.shape[1] + 1) / 2
    y_mid = (X_train_2d.shape[2] + 1) / 2
    center = np.array((x_mid, y_mid))

    for img in arr:
        x = 1
        new_img = []
        for row in img:
            y = 1
            # x_colored = 0
            distance_x = (pow(center[0] - x, 2))
            # print("dist x:", center[0] - x)
            for col in row:
                # y_colored = 0
                if col > 0:
                    distance_y = (pow(center[1] - y, 2))
                    # y_colored = 1
                    new_img.append(round(math.sqrt(distance_x + distance_y)))  # , y_colored))
                else:
                    new_img.append(0)
                y += 1
            x += 1
        if sort is True:
            euclid_array.append(np.sort(new_img)[::-1])
        else:
            euclid_array.append(new_img)
    return euclid_array


def add_euclidean_dist_count(arr):
    new_list = []
    for img in arr:
        for num in img:
            new_list.append(num)
            new_list.append(list(img).count(num))
        for i in range(32):
            new_list.append(0)
    new_arr = np.array(new_list).reshape(len(arr), 1600)
    return new_arr




# rowblur (testdata and traindata is 2 - dimensional)
def rowblur(testdata, traindata):
    testdata = (testdata != 0).astype(float)
    traindata = (traindata != 0).astype(float)

    trainlist = []
    testlist = []

    for i in range(testdata.shape[0]):
        buffer_list = []
        for j in range(testdata.shape[1]):
            buffer_list.append(np.sum(testdata[i, j]))
        testlist.append(buffer_list)

    for i in range(traindata.shape[0]):
        buffer_list = []
        for j in range(traindata.shape[1]):
            buffer_list.append(np.sum(traindata[i, j]))
        trainlist.append(buffer_list)

    return np.array(testlist), np.array(trainlist)


# columnblur (testdata and traindata is 2 - dimensional)
def columnblur(testdata, traindata):
    testdata = (testdata != 0).astype(float)
    traindata = (traindata != 0).astype(float)

    trainlist = []
    testlist = []

    for i in range(testdata.shape[0]):
        buffer_list = []
        for j in range(testdata.shape[1]):
            buffer_list.append(np.sum(testdata[i, :, j]))
        testlist.append(buffer_list)

    for i in range(traindata.shape[0]):
        buffer_list = []
        for j in range(traindata.shape[1]):
            buffer_list.append(np.sum(traindata[i, :, j]))
        trainlist.append(buffer_list)

    return np.array(testlist), np.array(trainlist)


# ALL FUNCTIONS NEEDED : row column blur is used for adding all rows and columns together
# Das ist der Zeilen, Spaltenfilter im Paper erwähnt
def columnblur_list(testdata, traindata):
    testdata = (testdata != 0).astype(float)
    traindata = (traindata != 0).astype(float)

    trainlist = []
    testlist = []

    for i in range(testdata.shape[0]):
        buffer_list = []
        for j in range(testdata.shape[1]):
            buffer_list.append(np.sum(testdata[i, :, j]))
        testlist.append(buffer_list)

    for i in range(traindata.shape[0]):
        buffer_list = []
        for j in range(traindata.shape[1]):
            buffer_list.append(np.sum(traindata[i, :, j]))
        trainlist.append(buffer_list)

    return testlist, trainlist


def rowblur_list(testdata, traindata):
    testdata = (testdata != 0).astype(float)
    traindata = (traindata != 0).astype(float)

    trainlist = []
    testlist = []

    for i in range(testdata.shape[0]):
        buffer_list = []
        for j in range(testdata.shape[1]):
            buffer_list.append(np.sum(testdata[i, j]))
        testlist.append(buffer_list)

    for i in range(traindata.shape[0]):
        buffer_list = []
        for j in range(traindata.shape[1]):
            buffer_list.append(np.sum(traindata[i, j]))
        trainlist.append(buffer_list)

    return testlist, trainlist


def row_column_blur(testdata, traindata):
    row_test, row_train = rowblur_list(testdata, traindata)
    column_test, column_train = columnblur_list(testdata, traindata)
    row_column_test = []
    row_column_train = []
    for i in range(row_test.__len__()):
        row_column_test.append(row_test[i] + column_test[i])
    for j in range(row_train.__len__()):
        row_column_train.append(row_train[j] + column_train[j])

    # TODO sort

    for i in range(row_column_test.shape[0]):
        row_column_test[i] = np.sort(row_column_test[i])

    for j in range(row_column_train.shape[0]):
        row_column_train[j] = np.sort(row_column_train[j])


    return np.array(row_column_test), np.array(row_column_train)




# takes in testdata and traindata as 2 dimensional pixel matrices --> returns filter values (filter used: circle)
# size_big == 1 uses (4,4) circle else (3,3) ----> generally higher accuracy with bigger circle
def circlefilter(testdata, traindata):
    size_big = 1
    test_list = []
    train_list = []

    if(size_big == 1):
        filter_shape = 4
        filter = np.zeros((4, 4)) - 1
        filter[0] = 1
        filter[3] = 1
        filter[:, 0] = 1
        filter[:, 3] = 1

    else:
        filter_shape = 3
        filter = np.zeros((3,3)) - 1
        filter[0] = 1
        filter[2] = 1
        filter[:, 0] = 1
        filter[:, 2] = 1

    for i in range(traindata.shape[0]):
        buffer_list = []
        for j in range(traindata.shape[1] - filter_shape):
            for k in range(traindata.shape[2] - filter_shape):
                x = 0
                for l in range(filter_shape):
                    for m in range(filter_shape):
                        x += traindata[i, j + m, k + l] * filter[m, l]
                buffer_list.append(x)
        train_list.append(buffer_list)

    for i in range(testdata.shape[0]):
        buffer_list = []
        for j in range(testdata.shape[1] - filter_shape):
            for k in range(testdata.shape[2] - filter_shape):
                x = 0
                for l in range(filter_shape):
                    for m in range(filter_shape):
                        x += testdata[i, j + m, k + l] * filter[m, l]
                buffer_list.append(x)
        test_list.append(buffer_list)

    return np.array(test_list), np.array(train_list)



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
    testdata = (testdata != 0)
    traindata = (traindata != 0)

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
X_train_2d = train_df.drop("label", axis=1).values.reshape(60000, 28,28)
X_test_2d = test_df.drop("label", axis=1).values.reshape(10000, 28,28)
X_train_1d = X_train_2d.reshape(60000, 784)
X_test_1d = X_test_2d.reshape(10000, 784)
y_train = train_df["label"]
y_test = test_df["label"]

methods = [naive_bayes.GaussianNB, naive_bayes.BernoulliNB]






############ AUSWERTUNG #############
"""
### SHIFT DEFAULT - 3 Pixel ###
test_shift = shift(X_test_2d[:2000], scale=3)
test_shift = np.array(test_shift).reshape(2000, 784)
train_test_data("Shift 3 Pixel", methods, X_train_1d[:10000], y_train[:10000], test_shift, y_test[:2000])

### EUKLID DISTANZ
train_distances = get_euclidean_distance(X_train_2d[:10000], sort=True)
test_shift = shift(X_test_2d[:2000])
test_shift_distances = get_euclidean_distance(test_shift[:2000], sort=True)
train_test_data("Shift 3 Pixel EUKLID", methods, train_distances, y_train[:10000], test_shift_distances, y_test[:2000])

### EUKLID DISTANZ + COUNT
train_shift_distances2 = add_euclidean_dist_count(train_distances)
test_shift_distances2 = add_euclidean_dist_count(test_shift_distances)
train_test_data("Shift 3 Pixel EUKLID + DISTANZ COUNT", methods, train_shift_distances2, y_train[:10000], test_shift_distances2,
                y_test[:2000])
"""
### FILTER
### ROW FILTER
test_shift = shift(X_test_2d[:2000], scale=3)
test_shift = np.array(test_shift)  # .reshape(2000, 784)
test_row_filter, train_row_filter = rowblur(test_shift, X_train_2d[:10000])
train_test_data("Shift 3 Pixel ROW FILTER", [naive_bayes.GaussianNB, naive_bayes.BernoulliNB], train_row_filter,
                y_train[:10000], test_row_filter, y_test[:2000])
"""
### COL FILTER
test_col_filter, train_col_filter = columnblur(test_shift, X_train_2d[:10000])
train_test_data("Shift 3 Pixel COL FILTER", [naive_bayes.GaussianNB, naive_bayes.BernoulliNB], train_col_filter,
                y_train[:10000], test_col_filter, y_test[:2000])

### ROW+COL FILTER
test_row_col_filter, train_row_col_filter = row_column_blur(test_shift, X_train_2d[:10000])
train_test_data("Shift 3 Pixel ROW+COL FILTER", [naive_bayes.GaussianNB, naive_bayes.BernoulliNB], train_row_col_filter,
                y_train[:10000], test_row_col_filter, y_test[:2000])


### KREISFILTER
test_rotated_circlefilter, train_rotated_circlefilter = circlefilter(test_shift, X_train_2d[:10000])
train_test_data("Shift 3 Pixel KREISFILTER", [naive_bayes.GaussianNB, naive_bayes.BernoulliNB], train_rotated_circlefilter, 
                y_train[:10000], test_rotated_circlefilter, y_test[:2000])


### Zoom in
test_shift = shift(X_test_2d[:2000], scale=3)
X_train_crop = zoom_in(X_train_2d[:10000]).reshape(10000, 400)
X_test_shift_crop = zoom_in(test_shift).reshape(2000, 400)
train_test_data("Shift 3 Pixel ZOOM IN", methods, X_train_crop, y_train[:10000], X_test_shift_crop, y_test[:2000])






### SHIFT DEFAULT - 6 Pixel ###
test_shift = shift(X_test_2d[:2000], scale=6)
test_shift = np.array(test_shift).reshape(2000, 784)
train_test_data("Shift 6 Pixel", methods, X_train_1d[:10000], y_train[:10000], test_shift, y_test[:2000])

### EUKLID DISTANZ
train_distances = get_euclidean_distance(X_train_2d[:10000], sort=True)
test_shift = shift(X_test_2d[:2000])
test_shift_distances = get_euclidean_distance(test_shift[:2000], sort=True)
train_test_data("Shift 6 Pixel EUKLID", methods, train_distances, y_train[:10000], test_shift_distances, y_test[:2000])

### EUKLID DISTANZ + COUNT
train_shift_distances2 = add_euclidean_dist_count(train_distances)
test_shift_distances2 = add_euclidean_dist_count(test_shift_distances)
train_test_data("Shift 6 Pixel EUKLID + DISTANZ COUNT", methods, train_shift_distances2, y_train[:10000], test_shift_distances2,
                y_test[:2000])
"""
### FILTER
### ROW FILTER
test_shift = shift(X_test_2d[:2000], scale=6)
test_shift = np.array(test_shift)  # .reshape(2000, 784)
test_row_filter, train_row_filter = rowblur(test_shift, X_train_2d[:10000])
train_test_data("Shift 6 Pixel ROW FILTER", [naive_bayes.GaussianNB, naive_bayes.BernoulliNB], train_row_filter,
                y_train[:10000], test_row_filter, y_test[:2000])
"""
### COL FILTER
test_col_filter, train_col_filter = columnblur(test_shift, X_train_2d[:10000])
train_test_data("Shift 6 Pixel COL FILTER", [naive_bayes.GaussianNB, naive_bayes.BernoulliNB], train_col_filter,
                y_train[:10000], test_col_filter, y_test[:2000])

### ROW+COL FILTER
test_row_col_filter, train_row_col_filter = row_column_blur(test_shift, X_train_2d[:10000])
train_test_data("Shift 6 Pixel ROW+COL FILTER", [naive_bayes.GaussianNB, naive_bayes.BernoulliNB], train_row_col_filter,
                y_train[:10000], test_row_col_filter, y_test[:2000])


### KREISFILTER
test_rotated_circlefilter, train_rotated_circlefilter = circlefilter(test_shift, X_train_2d[:10000])
train_test_data("Shift 6 Pixel KREISFILTER", [naive_bayes.GaussianNB, naive_bayes.BernoulliNB], train_rotated_circlefilter, 
                y_train[:10000], test_rotated_circlefilter, y_test[:2000])


### Zoom in
test_shift = shift(X_test_2d[:2000], scale=6)
X_train_crop = zoom_in(X_train_2d[:10000]).reshape(10000, 400)
X_test_shift_crop = zoom_in(test_shift).reshape(2000, 400)
train_test_data("Shift 6 Pixel ZOOM IN", methods, X_train_crop, y_train[:10000], X_test_shift_crop, y_test[:2000])





### SHIFT VERTIKAL DEFAULT - 6 Pixel ###
test_shift = shift(X_test_2d[:2000], scale=6, vertical=True)
test_shift = np.array(test_shift).reshape(2000, 784)
train_test_data("TEST Shift 6 Pixel VERTIKAL -", methods, X_train_1d[:10000], y_train[:10000], test_shift, y_test[:2000])

### EUKLID DISTANZ
train_distances = get_euclidean_distance(X_train_2d[:10000], sort=True)
test_shift = shift(X_test_2d[:2000])
test_shift_distances = get_euclidean_distance(test_shift[:2000], sort=True)
train_test_data("Shift 6 Pixel VERTIKAL - EUKLID", methods, train_distances, y_train[:10000], test_shift_distances, y_test[:2000])

### EUKLID DISTANZ + COUNT
train_shift_distances2 = add_euclidean_dist_count(train_distances)
test_shift_distances2 = add_euclidean_dist_count(test_shift_distances)
train_test_data("Shift 6 Pixel VERTIKAL - EUKLID + DISTANZ COUNT", methods, train_shift_distances2, y_train[:10000], test_shift_distances2,
                y_test[:2000])

### FILTER
### ROW FILTER
test_shift = shift(X_test_2d[:2000], scale=6, vertical=True)
test_shift = np.array(test_shift)  # .reshape(2000, 784)
test_row_filter, train_row_filter = rowblur(test_shift, X_train_2d[:10000])
train_test_data("Shift 6 Pixel VERTIKAL - ROW FILTER", [naive_bayes.GaussianNB, naive_bayes.BernoulliNB], train_row_filter,
                y_train[:10000], test_row_filter, y_test[:2000])

### COL FILTER
test_col_filter, train_col_filter = columnblur(test_shift, X_train_2d[:10000])
train_test_data("Shift 6 Pixel VERTIKAL - COL FILTER", [naive_bayes.GaussianNB, naive_bayes.BernoulliNB], train_col_filter,
                y_train[:10000], test_col_filter, y_test[:2000])

### ROW+COL FILTER
test_row_col_filter, train_row_col_filter = row_column_blur(test_shift, X_train_2d[:10000])
train_test_data("Shift 6 Pixel VERTIKAL - ROW+COL FILTER", [naive_bayes.GaussianNB, naive_bayes.BernoulliNB], train_row_col_filter,
                y_train[:10000], test_row_col_filter, y_test[:2000])


### KREISFILTER
test_rotated_circlefilter, train_rotated_circlefilter = circlefilter(test_shift, X_train_2d[:10000])
train_test_data("Shift 6 Pixel VERTIKAL - KREISFILTER", [naive_bayes.GaussianNB, naive_bayes.BernoulliNB], train_rotated_circlefilter, 
                y_train[:10000], test_rotated_circlefilter, y_test[:2000])


### Zoom in
test_shift = shift(X_test_2d[:2000], scale=6, vertical=True)
X_train_crop = zoom_in(X_train_2d[:10000]).reshape(10000, 400)
X_test_shift_crop = zoom_in(test_shift).reshape(2000, 400)
train_test_data("TEST Shift 6 Pixel VERTIKAL - ZOOM IN", methods, X_train_crop, y_train[:10000], X_test_shift_crop, y_test[:2000])



### Rotation Performance NO Euclidean NAIVE , angle=180 ###
test_shift = shift(X_test_2d[:2000], scale=6, vertical=True)
test_rotated = np.array(test_shift).reshape(2000, 784)
test, train = count(test_rotated[:2000], X_train_1d[:10000])
train_test_data("Shift VERTIAL 6 - NAIVE", methods, train[:10000], y_train[:10000], test, y_test[:2000])
"""
"""
### SHIFT DEFAULT - 3 Pixel ###
norm = X_test_2d[2]
test_shift3 = shift(X_test_2d[:2].copy(), scale=3)
test_shift6 = shift(X_test_2d[:2].copy(), scale=6)
test_shiftv6 = shift(X_test_2d[:2].copy(), scale=6, vertical=True)


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(5, 3))
plt.imshow(norm, cmap="gray", axes=0)
plt.imshow(test_shift3, cmap="gray", axes=1)
plt.imshow(test_shift6, cmap="gray", axes=2)
fig.tight_layout()
"""