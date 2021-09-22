import math
import cv2
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.ndimage import rotate
from sklearn import naive_bayes
from sklearn.metrics import classification_report



def train_test_data(mod:str,
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
        with open (path, "a") as f:
            f.write(f"\n\n {mod}:\n {model_name}: " + str(classification_report(y_test, pred, zero_division=0)))

        print("\n\n")
     

def rotation(cpytest, angle=90):
    """
    Returns 2D images
    """
    rotated = []
    for i in range(cpytest.shape[0]):
        rotated.append(rotate(cpytest[i], angle))
    return rotated



def get_euclidean_distance(arr, sort=False):
    """
    Function that transforms colored pixels to Euclidean distances
    """
    euclid_array = []
    x_mid = (X_train_2d.shape[1]+1)/2
    y_mid = (X_train_2d.shape[2]+1)/2
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
                    new_img.append(round(math.sqrt(distance_x+distance_y))) #, y_colored))
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





def columnblur_list(testdata, traindata):
    testdata = (testdata != 0)
    traindata = (traindata != 0)

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
    testdata = (testdata != 0)
    traindata = (traindata != 0)

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
    row_column_test = np.array(row_column_test)
    row_column_train = np.array(row_column_train)

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


def sort(row_column_test, row_column_train):
    for i in range(row_column_test.shape[0]):
        row_column_test[i] = np.sort(row_column_test[i])

    for j in range(row_column_train.shape[0]):
        row_column_train[j] = np.sort(row_column_train[j])

    return np.array(row_column_test), np.array(row_column_train)


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

### Rotation Performance NO Euclidean , angle=90 ###
test_rotated = rotation(X_test_2d[:2000], angle=90)
import matplotlib.pyplot as plt
plt.imshow(test_rotated[4], cmap="gray")
plt.show()
test_rotated = np.array(test_rotated).reshape(2000, 784)
# train_test_data("Rotation 90", methods, X_train_1d[:10000], y_train[:10000], test_rotated, y_test[:2000])
"""
### ZEILENFILTER
test_rotated_row_filter, train_row_filter = rowblur_list(test_rotated, X_train_1d[:10000])
test_rotated_row_filter = np.array(test_rotated_row_filter).reshape(2000, 784)
train_row_filter = np.array(train_row_filter).reshape(10000, 784)
train_test_data("Rotation 90 - ROW FILTER", methods, train_row_filter, y_train[:10000], test_rotated_row_filter, y_test[:2000])

### SPALTENFILTER
test_rotated = test_rotated.reshape(2000, 28, 28)
test_rotated_col_filter, train_col_filter = columnblur_list(test_rotated, X_train_2d[:10000])
# test_rotated_col_filter = np.array(test_rotated_col_filter).reshape(2000, 784)
# train_col_filter = np.array(train_col_filter).reshape(10000, 784)
train_test_data("Rotation 90 - COL FILTER", methods, train_col_filter, y_train[:10000], test_rotated_col_filter, y_test[:2000])

### ZEILEN + SPALTENFILTER
test_rotated = test_rotated.reshape(2000, 28, 28)
test_rotated_col_filter, train_col_filter = row_column_blur(test_rotated, X_train_2d[:10000])
# test_rotated_col_filter = np.array(test_rotated_col_filter).reshape(2000, 784)
# train_col_filter = np.array(train_col_filter).reshape(10000, 784)
train_test_data("Rotation 90 - COL ROW SORTED ", methods, train_col_filter, y_train[:10000], test_rotated_col_filter, y_test[:2000])

### EUCLIDEAN
test_rotated = rotation(X_test_2d[:2000], angle=90)
train_distances = np.array(get_euclidean_distance(X_train_2d[:10000], sort=True))
test_rotated_distances = np.array(get_euclidean_distance(test_rotated, sort=True))
train_test_data("Rotation 90 - EUKLID", methods, train_distances, y_train[:10000], test_rotated_distances, y_test[:2000])

### ROTATION ADD DISTANCES COUNT
train_distances2 = add_euclidean_dist_count(train_distances)
test_rotated_distances2 = add_euclidean_dist_count(test_rotated_distances)
train_test_data("Rotation 90 - EUKLID + DISTANZ", methods, train_distances2, y_train[:10000], test_rotated_distances2, y_test[:2000])

### Zoom in
X_train_crop = zoom_in(X_train_2d[:10000]).reshape(10000, 400)
X_test_rotated_crop = zoom_in(test_rotated[:2000]).reshape(2000, 400)
train_test_data("Rotation 90 - ZOOM IN", methods, X_train_crop, y_train[:10000], X_test_rotated_crop, y_test[:2000])

### KREISFILTER
test_rotated = rotation(X_test_2d[:2000], angle=90)
test_rotated = np.array(test_rotated) #.reshape(20, 784)
test_rotated_circlefilter, train_rotated_circlefilter = circlefilter(test_rotated, X_train_2d[:10000])
test_rotated_circlefilter, train_rotated_circlefilter = sort(test_rotated_circlefilter, train_rotated_circlefilter)
train_test_data("Rotation 90 - KREISFILTER", [naive_bayes.GaussianNB, naive_bayes.BernoulliNB], train_rotated_circlefilter, y_train[:10000], test_rotated_circlefilter, y_test[:2000])




### Rotation Performance NO Euclidean , angle=180 ###
test_rotated = rotation(X_test_2d[:2000], angle=180)
test_rotated = np.array(test_rotated).reshape(2000, 784)
train_test_data("Rotation 180", methods, X_train_1d[:10000], y_train[:10000], test_rotated, y_test[:2000])

### ZEILENFILTER
test_rotated_row_filter, train_row_filter = rowblur_list(test_rotated, X_train_1d[:10000])
test_rotated_row_filter = np.array(test_rotated_row_filter).reshape(2000, 784)
train_row_filter = np.array(train_row_filter).reshape(10000, 784)
train_test_data("Rotation 180 - ROW FILTER", methods, train_row_filter, y_train[:10000], test_rotated_row_filter, y_test[:2000])

### SPALTENFILTER
test_rotated = test_rotated.reshape(2000, 28, 28)
test_rotated_col_filter, train_col_filter = columnblur_list(test_rotated, X_train_2d[:10000])
# test_rotated_col_filter = np.array(test_rotated_col_filter).reshape(2000, 784)
# train_col_filter = np.array(train_col_filter).reshape(10000, 784)
train_test_data("Rotation 180 - COL FILTER", methods, train_col_filter, y_train[:10000], test_rotated_col_filter, y_test[:2000])

### ZEILEN + SPALTENFILTER
test_rotated = rotation(X_test_2d[:2000], angle=180)
test_rotated = test_rotated.reshape(2000, 28, 28)
test_rotated_col_filter, train_col_filter = row_column_blur(test_rotated, X_train_2d[:10000])
# test_rotated_col_filter = np.array(test_rotated_col_filter).reshape(2000, 784)
# train_col_filter = np.array(train_col_filter).reshape(10000, 784)
train_test_data("Rotation 180 - COL ROW SORTED ", methods, train_col_filter, y_train[:10000], test_rotated_col_filter, y_test[:2000])

### EUCLIDEAN
test_rotated = rotation(X_test_2d[:2000], angle=180)
train_distances = np.array(get_euclidean_distance(X_train_2d[:10000], sort=True))
test_rotated_distances = np.array(get_euclidean_distance(test_rotated, sort=True))
train_test_data("Rotation 180 - EUKLID", methods, train_distances, y_train[:10000], test_rotated_distances, y_test[:2000])

### ROTATION ADD DISTANCES COUNT
train_distances2 = add_euclidean_dist_count(train_distances)
test_rotated_distances2 = add_euclidean_dist_count(test_rotated_distances)
train_test_data("Rotation 180 - EUKLID + DISTANZ", methods, train_distances2, y_train[:10000], test_rotated_distances2, y_test[:2000])

### Zoom in
X_train_crop = zoom_in(X_train_2d[:10000]).reshape(10000, 400)
X_test_rotated_crop = zoom_in(test_rotated[:2000]).reshape(2000, 400)
train_test_data("Rotation 180 - ZOOM IN", methods, X_train_crop, y_train[:10000], X_test_rotated_crop, y_test[:2000])

### KREISFILTER
test_rotated = rotation(X_test_2d[:2000], angle=180)
test_rotated = np.array(test_rotated) #.reshape(20, 784)
test_rotated_circlefilter, train_rotated_circlefilter = circlefilter(test_rotated, X_train_2d[:10000])
test_rotated_circlefilter, train_rotated_circlefilter = sort(test_rotated_circlefilter, train_rotated_circlefilter)
train_test_data("Rotation 180 - KREISFILTER", [naive_bayes.GaussianNB, naive_bayes.BernoulliNB], train_rotated_circlefilter, y_train[:10000], test_rotated_circlefilter, y_test[:2000])




### NAIVE ### 
### Rotation Performance NO Euclidean , angle=90 ###
test_rotated = rotation(X_test_2d[:2000], angle=90)
test_rotated = np.array(test_rotated).reshape(2000, 784)
test, train = count(test_rotated[:2000], X_train_1d[:10000])
train_test_data("Rotation 90 - NAIVE", methods, train[:10000], y_train[:10000], test, y_test[:2000])

### Rotation Performance NO Euclidean , angle=180 ###
test_rotated = rotation(X_test_2d[:2000], angle=180)
test_rotated = np.array(test_rotated).reshape(2000, 784)
test, train = count(test_rotated[:2000], X_train_1d[:10000])
train_test_data("Rotation 120 - NAIVE", methods, train[:10000], y_train[:10000], test, y_test[:2000])
"""