import matplotlib.pyplot as plt
import numpy as np

def f(x):
    x = x.ravel()
    return np.exp(-(x ** 2)) + 1.5 * np.exp(-((x - 2) ** 2))

def plot_target(X_train, y_train, X_test):

    plt.figure(figsize=(10, 6))
    plt.plot(X_test, f(X_test), "b")
    plt.scatter(X_train, y_train, c="b", s=20)
    plt.xlim([-5, 5])
    plt.show()
