import numpy as np
from numpy import linalg as la


def lstsq_ne(A: np.ndarray, b: np.ndarray) -> tuple:
    inv = la.inv(np.dot(A.T, A))
    x = np.dot(inv, np.dot(A.T, b))
    y = np.dot(A, x) - b
    cost = np.dot(y.T, y)
    var = inv * cost / (A.shape[0] - A.shape[1])
    return x, cost, var


def lstsq_svd(A: np.ndarray, b: np.ndarray, rcond: float = None) -> tuple:
    U, s, Vh = la.svd(A, full_matrices=False)
    if rcond:
        s[s < rcond * s[0]] = 0.
    s_inv = np.diag(np.divide(1, s, where=s != 0.))
    x = np.dot(Vh.T, np.dot(s_inv, np.dot(U.T, b)))
    y = np.dot(A, x) - b
    cost = np.dot(y.T, y)
    var = np.dot(Vh.T, np.dot(np.diag(np.divide(1, s ** 2, where=s != 0.)
                                      ), Vh)) * cost / (A.shape[0] -A.shape[1])
    return x, cost, var


def lstsq(A: np.ndarray, b: np.ndarray, method='ne', **kwargs) -> tuple:
    if method.lower() == 'ne':
        return lstsq_ne(A, b, **kwargs)
    if method.lower() == 'svd':
        return lstsq_svd(A, b, **kwargs)
    raise NotImplementedError(method)
