import numpy as np
from keras.datasets import mnist
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from datetime import datetime

#from numba import jit, cuda
#import code


class NaiveBayes:
    def __init__(self):
        self.gauss = dict()
        self.a_priori_class = dict()

    def fit(self, X, y):
        labels = y
        for l in labels:
            x = X[y == l]
            self.gauss[l] = {"mean": x.mean(axis=0),
                             "variance": (x.var(axis=0) + 10e-3)
                             }
            # Berechne a priori Wahrscheinlichkeit für jedes Label
            self.a_priori_class[l] = len(y[y == l]) / len(y)
            # print(f"P({l}): {self.a_priori_class[l]:.2f}")

    def predict(self, X):
        n, m = X.shape
        k = len(self.gauss)
        P = np.zeros((n, k))
        for c, g in self.gauss.items():
            mean, variance = g["mean"], g["variance"]
            P[:, c] = mvn.logpdf(X, mean=mean, cov=variance) + np.log(self.a_priori_class[c])
        return np.argmax(P, axis=1)

    def score(self, X, y):
        P = self.predict(X)
        return np.mean(P == y)


if __name__ == "__main__":
    # Lade MNIST Daten
    (X_train, y_train),(X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((60000, 784))
    X_test = X_test.reshape((10000, 784))

    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    t0 = datetime.now()
    print("Accuracy on Training Data:    ", nb.score(X_train, y_train))
    print("     Train Size:       ", len(y_train))
    print("     Time to complete: ", (datetime.now() - t0))
    print()
    print()
    t1 = datetime.now()
    print("Accuracy on Test Data:        ", nb.score(X_test, y_test))
    print("     Test Size:        ", len(y_test))
    print("     Time to complete: ", (datetime.now() - t1))
