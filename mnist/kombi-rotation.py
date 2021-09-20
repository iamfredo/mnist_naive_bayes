import numpy as np
import scipy as sp
import sklearn
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import pandas as pd
import cv2
import math
from sklearn import naive_bayes


# Einlesen der Daten mit pandas
dftest = pd.read_csv(r'../data/mnist_test.csv', nrows=2000)
dftrain = pd.read_csv(r'../data/mnist_train.csv', nrows=10000)

# Slicen der Testdaten
testsl = np.zeros((dftest.values.shape[0], dftest.values.shape[1] - 1))
for i in range(dftest.values.shape[0]):
    testsl[i] = dftest.values[i][1:]

testlabels = np.zeros(dftest.values.shape[0])

for i in range(dftest.values.shape[0]):
    testlabels[i] = dftest.values[i][0]

# Slicen der Trainingsdaten

trainsl = np.zeros((dftrain.values.shape[0], dftrain.values.shape[1] - 1))
for i in range(dftrain.values.shape[0]):
    trainsl[i] = dftrain.values[i][1:]

trainlabels = np.zeros(dftrain.values.shape[0])

for i in range(dftrain.values.shape[0]):
    trainlabels[i] = dftrain.values[i][0]


# Trainingsdaten

cpytrain = trainsl.reshape(trainsl.shape[0], np.sqrt(trainsl.shape[1]).astype(int),
                           np.sqrt(trainsl.shape[1]).astype(int))
# Testdaten
cpytest = testsl.reshape(testsl.shape[0], np.sqrt(testsl.shape[1]).astype(int), np.sqrt(testsl.shape[1]).astype(int))



#go

for i in range(testsl.shape[0]):
    cpytest[i] = rotate(cpytest[i], 180, order = 0)
testsl = cpytest.reshape(testsl.shape[0], testsl.shape[1])


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


    for i in range(row_column_test.__len__()):
        row_column_test[i] = np.sort(row_column_test[i])
    for i in range(row_column_train.__len__()):
        row_column_train[i] = np.sort(row_column_train[i])


    return row_column_test, row_column_train


def get_euclidian_distance(arr, sort=False):
    """
    Function that transforms colored pixels to euclidian distances
    """
    center = [14.5, 14.5]
    euclid_array = []

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


testsl1 , trainsl1 = row_column_blur(cpytest, cpytrain)
testsl = get_euclidian_distance(cpytest, sort = True)
trainsl = get_euclidian_distance(cpytrain, sort = True)
new_test = []
new_train = []

for i in range(testsl.__len__()):
    new_test.append(list(testsl[i]) + list(testsl1[i]))

for i in range(trainsl.__len__()):
    new_train.append(list(trainsl[i]) + list(trainsl1[i]))

testsl = new_test
trainsl = new_train
# Create a classifier: a support vector classifier
classifier = naive_bayes.GaussianNB()


#sort (Pixel als Menge betrachten)

#for i in range(testsl.shape[0]):
#    testsl[i] = np.sort(testsl[i])

#for j in range(trainsl.shape[0]):
#    trainsl[j] = np.sort(trainsl[j])


# Trainiere Daten
classifier.fit(trainsl, trainlabels)  # Berechnet die Hyperebene/Modell

# Man verschiebe das Bild um 20 Pixel nach oben


# TODO

# Predict the value of the digit on the test subject


predicted = classifier.predict(testsl)  # Man wende das Modell auf die Testdaten an


print(f"Classification report for classifier {classifier}:\n"
      f"{metrics.classification_report(testlabels, predicted)}\n")

'TRASH'

# for i in range(testsl.shape[0]):
#    cpytest[i] = rotate(cpytest[i], 90)
# testsl = cpytest.reshape(testsl.shape[0], testsl.shape[1])


# Normieren der Test- und Trainingsdaten



#sort

#for i in range(testsl.shape[0]):
#    testsl[i] = np.sort(testsl[i])

#for j in range(trainsl.shape[0]):
#    trainsl[j] = np.sort(trainsl[j])