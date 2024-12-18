import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

class BregmanADMM(nn.Module):
    def __init__(self, rho, lambda_, alpha, n_it):
        super(BregmanADMM, self).__init__()
        self.rho = rho
        self.lambda_ = lambda_
        self.alpha = alpha
        self.n_it = n_it

    def soft_threshold(self, x, threshold):
        return torch.sign(x) * torch.maximum(torch.abs(x) - threshold, torch.zeros_like(x))

    def compute_R(self, R_0, X_1, X_2, Z_1, Z_2, rho):
        min_value = torch.tensor(1e-40, dtype=torch.float64)
        R_0 = R_0.cuda()
        min_value = min_value.cuda()
        R_0 = torch.where((R_0 <= 0), min_value, R_0)
        X_1 = torch.where((X_1 <= 0), 1e-40, X_1)
        X_2 = torch.where((X_2 <= 0), min_value, X_2)

        tmp = (torch.log(R_0) + rho * torch.log(X_1) + rho * torch.log(X_2) - rho * Z_1 - rho * Z_2) / (1 + 2 * rho)
        R = torch.softmax(tmp, dim=2)

        return R

    def compute_X1(self, R, Z_1, lambda_, alpha, rho):
        lambda1 = lambda_ * alpha
        X_1 = self.soft_threshold(R + Z_1, lambda1 / rho)
        return X_1

    def compute_X2(self, R, Z_2, lambda_, alpha, rho):
        lambda2 = lambda_ * (1 - alpha)
        identity_matrix = 1e-5 * torch.eye(R.shape[2], device=R.device, dtype=R.dtype)
        identity_matrix = identity_matrix.unsqueeze(0).repeat(R.shape[0], 1, 1)
        adjusted_matrix = R + Z_2 + identity_matrix

        U, S, Vt = torch.linalg.svd(adjusted_matrix, full_matrices=False)
        X_2 = U @ torch.diag_embed(self.soft_threshold(S, lambda2 / rho)) @ Vt
        X_2 = torch.where(X_2 < 0, 1e-8, X_2)
        X_2 = X_2.detach()
        return X_2

    def compute_Z(self, Z, R, X):
        Z = Z + R - X
        return Z

    def forward(self, R_0):
        R = R_0
        X_1 = R_0
        X_2 = R_0
        Z_1 = torch.zeros_like(R_0)
        Z_2 = torch.zeros_like(R_0)

        for t in range(self.n_it):
            R = self.compute_R(R_0, X_1, X_2, Z_1, Z_2, self.rho)
            X_1 = self.compute_X1(R, Z_1, self.lambda_, self.alpha, self.rho)
            X_2 = self.compute_X2(R, Z_2, self.lambda_, self.alpha, self.rho)
            Z_1 = self.compute_Z(Z_1, R, X_1)
            Z_2 = self.compute_Z(Z_2, R, X_2)
        # print("R:\n",R)

        return R
