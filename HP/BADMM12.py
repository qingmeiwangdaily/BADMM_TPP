import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

def softmax(x, axis=None):
    shiftx = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shiftx)
    sum_exps = np.sum(exps, axis=axis, keepdims=True)
    return exps / sum_exps

def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, np.zeros_like(x))

def compute_R(R_0, X, Z, rho):
    min_value = 1e-100  
    R_0 = np.where((R_0 == 0), min_value, R_0)
    X = np.where((X == 0), min_value, X)
    tmp = (np.log(R_0) + rho * np.log(X) - Z) / (1 + rho)
    R = softmax(tmp, axis=1)
    return R

def compute_X(R, Z, alpha, lambd, rho):
    X = np.zeros_like(R)
    for i in range(len(R)):
        s_td = soft_threshold(R[:, i] + 1. * Z[:, i] / rho, alpha * lambd / rho)
        X[:, i] = np.maximum(1 - (1. - alpha) * lambd / (rho * np.linalg.norm(s_td, ord=2)), 0) * s_td
    return X

def compute_Z(Z,R,X,rho):
    Z = Z + rho * (R - X)
    return Z

def BregmanADMM12(R_0,rho=1, alpha=0.1, lambd=0.1, num_iteration=10):
    R = R_0
    X = R_0
    Z = np.zeros_like(R_0)
    for t in range(num_iteration):
        R = compute_R(R_0, X, Z, rho)
        X = compute_X(R, Z, alpha, lambd, rho)
        Z = compute_Z(Z, R, X, rho)
    return R
