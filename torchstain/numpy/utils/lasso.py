import numpy as np


"""
LASSO implementation was adapted from:
https://www.kaggle.com/code/mcweng24/lasso-regression-using-numpy
"""
def predicted_values(X, w):
    return np.matmul(X, w)


def rho_compute(y, X, w, j):
    X_k = np.delete(X, j, 1)
    w_k = np.delete(w, j)
    predict_k = predicted_values(X_k, w_k)
    residual = y - predict_k
    return np.sum(X[:, j] * residual)


def z_compute(X):
    z_vector = np.sum(X * X, axis=0)
    return np.sum(X * X, axis = 0)


def coordinate_descent(y, X, w, alpha, z, tol):
    max_step = 100
    iteration = 0
    while max_step > tol:
        iteration += 1
        old_weights = np.copy(w)
        for j in range(len(w)):
            rho_j = rho_compute(y, X, w, j)
            if j == 0:
                w[j] = rho_j / z[j]
            elif rho_j < -alpha * len(y):
                w[j] = (rho_j + (alpha * len(y))) / z[j]
            elif rho_j > -alpha * len(y) and rho_j < alpha * len(y):
                w[j] = 0.
            elif rho_j > alpha * len(y):
                w[j] = (rho_j - (alpha * len(y))) / z[j]
            else:
                w[j] = np.NaN
        step_sizes = np.abs(old_weights - w)
        max_step = step_sizes.max()
    return w, iteration, max_step


def lasso(x, y, alpha=0.1, tol=0.0001):
    w = np.zeros(x.shape[1], dtype="float32")
    z = z_compute(x)
    w_opt, iterations, max_step = coordinate_descent(y, x, w, alpha, z, tol)
    return w_opt
