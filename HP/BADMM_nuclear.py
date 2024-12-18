import numpy as np

def softmax(x, axis=None):
    shiftx = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shiftx)
    sum_exps = np.sum(exps, axis=axis, keepdims=True)
    return exps / sum_exps

def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, np.zeros_like(x))

def compute_R(R_0, X_1, X_2, Z_1, Z_2, rho):
    min_value = 1e-40
    R_0 = np.where((R_0 <= 0), min_value, R_0)
    X_1 = np.where((X_1 <= 0), min_value, X_1)
    X_2 = np.where((X_2 <= 0), min_value, X_2)
    tmp = (np.log(R_0) + rho * np.log(X_1) + rho * np.log(X_2) - rho * Z_1 - rho * Z_2 ) / (1 + 2 * rho)
    R = softmax(tmp, axis=1)
    return R

def compute_X1(R, Z_1, lambd_,alpha, rho):
    lambd1 = lambd_ * alpha
    X_1 = soft_threshold(R + Z_1, lambd1 / rho)
    return X_1

def compute_X2(R, Z_2, lambda_, alpha, rho):
    lambd2 = lambda_ * (1 - alpha)
    identity_matrix = 1e-5 * np.eye(len(R))
    U, S, Vt = np.linalg.svd(R + Z_2 + identity_matrix, full_matrices=False)
    X_2 = U @ np.diag(soft_threshold(S, lambd2 / rho)) @ Vt
    X_2 = np.where(X_2 < 0, 1e-8, X_2)
    return X_2

def compute_Z(Z, R, X):
    Z = Z + R - X
    return Z

def BregmanADMM_nuclear(R_0,rho=1, lambd=0.6, alpha=0.02, num_iteration=10):
    R = R_0
    X_1 = R_0
    X_2 = R_0
    Z_1 = np.zeros_like(R_0)
    Z_2 = np.zeros_like(R_0)

    for t in range(num_iteration):
        R = compute_R(R_0, X_1, X_2, Z_1, Z_2, rho)
        X_1 = compute_X1(R, Z_1, lambd, alpha,rho)
        X_2 = compute_X2(R, Z_2, lambd, alpha,rho)
        Z_1 = compute_Z(Z_1, R, X_1)
        Z_2 = compute_Z(Z_2, R, X_2)
    return R
