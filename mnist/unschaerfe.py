from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn import naive_bayes
from sklearn.metrics import classification_report
from skimage.morphology import skeletonize


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
        print(f"{model_name}" + str(
            classification_report(y_test, pred, zero_division=0)))

        path = Path(f"../data/data_final/{mod}.txt")
        with open(path, "a") as f:
            f.write(f"\n\n {mod}:\n {model_name}: " + str(classification_report(y_test, pred, zero_division=0)))

        print("\n\n")


# Blurring der Testdaten
def blur(arr, size=(4, 4)):
    for i in range(arr.shape[0]):
        arr[i] = cv2.blur(arr[i], size)
    return arr

 
# skeletonization
def skel(cpytest, cpytrain):
    # for i in range(cpytest.shape[0]):
    #     cpytest[i] = cv2.blur(cpytest[i], (3, 3))
    cpytest = (cpytest != 0).astype(float)
    cpytrain = (cpytrain != 0).astype(float)
    for i in range(cpytrain.shape[0]):
        cpytrain[i] = skeletonize(cpytrain[i])
    for i in range(cpytest.shape[0]):
        cpytest[i] = skeletonize(cpytest[i])
    return cpytest, cpytrain




# ZOOM
def zoom_in(arr):
    my_arr = []
    for elem in arr:
        p = np.array(elem, dtype="uint8")
        # p = elem
        p = p.reshape((28,28))
        x,y,w,h = cv2.boundingRect(p)
        img1 = p[y:(y+h), x:(x+w)]
        imgResized = cv2.resize(img1,(20,20))
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






############ AUSWERTUNG #############

### BLUR DEFAULT 4x4
test_blur = blur(X_test_2d[:2000], size=(4,4))
test_blur = np.array(test_blur).reshape(2000, 784)
train_test_data("Blur 4x4", methods, X_train_1d[:10000], y_train[:10000], test_blur, y_test[:2000])

## BLUR SKELETON 4x4
test_blur = blur(X_test_2d[:2000], size=(4,4))
# test_blur = np.array(test_blur).reshape(2000, 784)
test_blur_skeleton, train_skeleton = skel(test_blur, X_test_2d[:10000])
train_skeleton = np.array(train_skeleton).reshape(10000, 784)
test_blur_skeleton = np.array(test_blur_skeleton).reshape(2000, 784)
train_test_data("Blur 4x4 - SKELETON", methods, train_skeleton[:10000], y_train[:10000], test_blur_skeleton, y_test[:2000])

### Zoom in 4x4
test_blur = blur(X_test_2d[:2000], size=(4,4))
X_train_crop = zoom_in(X_train_2d[:10000]).reshape(10000, 400)
X_test_blur_crop = zoom_in(test_blur).reshape(2000, 400)
train_test_data("Blur 4x4 -  ZOOM IN", methods, X_train_crop, y_train[:10000], X_test_blur_crop, y_test[:2000])

### BLUR EUCLIDEAN
train_distances = get_euclidean_distance(X_train_2d[:10000], sort=True)
test_blur_distances = get_euclidean_distance(test_blur[:2000], sort=True)
train_test_data("Blur 4x4 EUKLID", methods, train_distances, y_train[:10000], test_blur_distances, y_test[:2000])

### BLUR EUCLIDEAN + COUNT
train_distances2 = add_euclidean_dist_count(train_distances)
test_blur_distances2 = add_euclidean_dist_count(test_blur_distances)
train_test_data("Blur 4x4 EUKLID + DISTANZ COUNT", methods, train_distances2, y_train[:10000], test_blur_distances2, y_test[:2000])






### BLUR DEFAULT 5x5
test_blur = blur(X_test_2d[:2000], size=(5, 5))
test_blur = np.array(test_blur).reshape(2000, 784)
train_test_data("Blur 5x5", methods, X_train_1d[:10000], y_train[:10000], test_blur, y_test[:2000])

## BLUR SKELETON 5x5
test_blur = blur(X_test_2d[:2000], size=(5, 5))
# test_blur = np.array(test_blur).reshape(2000, 784)
test_blur_skeleton, train_skeleton = skel(test_blur, X_test_2d[:10000])
train_skeleton = np.array(train_skeleton).reshape(10000, 784)
test_blur_skeleton = np.array(test_blur_skeleton).reshape(2000, 784)
train_test_data("Blur 5x5 - SKELETON", methods, train_skeleton[:10000], y_train[:10000], test_blur_skeleton, y_test[:2000])

### Zoom in 5x5
test_blur = blur(X_test_2d[:2000], size=(5, 5))
X_train_crop = zoom_in(X_train_2d[:10000]).reshape(10000, 400)
X_test_blur_crop = zoom_in(test_blur).reshape(2000, 400)
train_test_data("Blur 5x5 -  ZOOM IN", methods, X_train_crop, y_train[:10000], X_test_blur_crop, y_test[:2000])






### BLUR DEFAULT 6x6
test_blur = blur(X_test_2d[:2000], size=(6, 6))
test_blur = np.array(test_blur).reshape(2000, 784)
train_test_data("Blur 6x6", methods, X_train_1d[:10000], y_train[:10000], test_blur, y_test[:2000])

## BLUR SKELETON 6x6
test_blur = blur(X_test_2d[:2000], size=(6, 6))
# test_blur = np.array(test_blur).reshape(2000, 784)
test_blur_skeleton, train_skeleton = skel(test_blur, X_test_2d[:10000])
train_skeleton = np.array(train_skeleton).reshape(10000, 784)
test_blur_skeleton = np.array(test_blur_skeleton).reshape(2000, 784)
train_test_data("Blur 6x6 - SKELETON", methods, train_skeleton[:10000], y_train[:10000], test_blur_skeleton, y_test[:2000])

### Zoom in 6x6
test_blur = blur(X_test_2d[:2000], size=(6, 6))
X_train_crop = zoom_in(X_train_2d[:10000]).reshape(10000, 400)
X_test_blur_crop = zoom_in(test_blur).reshape(2000, 400)
train_test_data("Blur 6x6 -  ZOOM IN", methods, X_train_crop, y_train[:10000], X_test_blur_crop, y_test[:2000])