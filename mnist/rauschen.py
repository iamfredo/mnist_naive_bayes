import numpy as np
import pandas as pd
from sklearn import naive_bayes
from pathlib import Path
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import cv2


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



def dirtyMNIST(number_array, scale=5):
      # Skalierungsvariable
    n = scale
    for i in range(number_array.shape[0]):
        random_matrix = np.zeros((28, 28))
        for j in range(n):
            z = np.random.randint(1, 255)
            x = np.random.randint(0, 28)
            y = np.random.randint(0, 28)
            random_matrix[x, y] += z
        number_array[i] = number_array[i] + random_matrix

    return number_array

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

    return np.array(row_column_test), np.array(row_column_train)




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






############### AUSWERTUNG ################
### Bildrauschen Default 100 Pixel ###
test_dirty = dirtyMNIST(X_test_2d[:2000], scale=100)
plt.imshow(test_dirty[0])
plt.show()
test_dirty = np.array(test_dirty).reshape(2000, 784)
#train_test_data("Bildrauschen 100 Pixel", methods, X_train_1d[:10000], y_train[:10000], test_dirty, y_test[:2000])

### Bildrauschen Blur Filter - ROW+COL
# Noise Injection on Test Data
test_dirty = dirtyMNIST(X_test_2d[:2000], scale=100)
test_dirty = np.array(test_dirty) #.reshape(20, 784)
test_blur_filter, train_blur_filter = row_column_blur(test_dirty, X_train_2d[:10000])
plt.imshow(test_blur_filter[0])
plt.show()
"""
train_test_data("Bildrauschen 100 Pixel - Blur Filter Row+Col", [naive_bayes.GaussianNB, naive_bayes.BernoulliNB], train_blur_filter, y_train[:10000], test_blur_filter, y_test[:2000])

### Zoom in
test_dirty = dirtyMNIST(X_test_2d[:2000], scale=100)
X_train_crop = zoom_in(X_train_2d[:10000]).reshape(10000, 400)
X_test_dirty_crop = zoom_in(test_dirty).reshape(2000, 400)
train_test_data("Bildrauschen 100 Pixel ZOOM IN", methods, X_train_crop, y_train[:10000], X_test_dirty_crop, y_test[:2000])



### Bildrauschen Default 200 Pixel ###
test_dirty = dirtyMNIST(X_test_2d[:2000], scale=200)
plt.imshow(test_dirty[0])
plt.show()
test_dirty = np.array(test_dirty).reshape(2000, 784)
train_test_data("Bildrauschen 200 Pixel", methods, X_train_1d[:10000], y_train[:10000], test_dirty, y_test[:2000])

### Bildrauschen Blur Filter - ROW+COL
# Noise Injection on Test Data
test_dirty = dirtyMNIST(X_test_2d[:2000], scale=200)
test_dirty = np.array(test_dirty) #.reshape(20, 784)
test_blur_filter, train_blur_filter = row_column_blur(test_dirty, X_train_2d[:10000])
train_test_data("Bildrauschen 200 Pixel - Blur Filter Row+Col", [naive_bayes.GaussianNB, naive_bayes.BernoulliNB], train_blur_filter, y_train[:10000], test_blur_filter, y_test[:2000])

### Zoom in
test_dirty = dirtyMNIST(X_test_2d[:2000], scale=200)
X_train_crop = zoom_in(X_train_2d[:10000]).reshape(10000, 400)
X_test_dirty_crop = zoom_in(test_dirty).reshape(2000, 400)
train_test_data("Bildrauschen 200 Pixel ZOOM IN", methods, X_train_crop, y_train[:10000], X_test_dirty_crop, y_test[:2000])
"""