import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

class BregmanADMM12(nn.Module):

    def __init__(self, rho,  lambd_, alpha, n_it):
        super(BregmanADMM12, self).__init__()
        self.rho = rho
        self.alpha = alpha
        self.lambd = lambd_
        self.n_it = n_it
    
    def forward(self, R_0):
        R = R_0
        X = R_0
        Z = torch.zeros_like(R_0)
        for t in range(self.n_it):
            R = self.compute_R(R_0, X, Z, self.rho)
            X = self.compute_X(R, Z, self.alpha, self.lambd, self.rho)
            Z = self.compute_Z(Z, R, X, self.rho)
            # print("\n\nnum_iteration:",t,"\nR:",R,"\nX:",X,"\nZ:",Z)
        return R

    def compute_R(self, R_0, X, Z, rho):
        # min_value = 1e-100  
        min_value = torch.tensor(1e-40, dtype=torch.float32)
        # R_0 = np.clip(R_0, min_value, None)
        # X = np.clip(X, min_value, None)
        R_0 = R_0.cuda()
        min_value = min_value.cuda()
        R_0 = torch.where((R_0 == 0), min_value, R_0)
        X = torch.where((X == 0), min_value, X)
        tmp = (torch.log(R_0) + rho * torch.log(X) - Z) / (1 + rho)
        R = torch.softmax(tmp, dim=2)
        return R

    def compute_X(self, R, Z, alpha, lambd, rho):
        X = torch.zeros_like(R)
        for i in range(R.shape[2]):
            x = R[:, :, i] + 1. * Z[:, :, i] / rho
            threshold = alpha * lambd / rho
            x = x.cuda()
            threshold = torch.ones_like(x) * threshold
            s_td = torch.sign(x) * torch.maximum(torch.abs(x) - threshold, torch.zeros_like(x))
            # s_td = soft_threshold(R[:, :, i] + 1. * Z[:, :, i] / rho, alpha * lambd / rho)
            epsilon = 1e-10
            X[:, :, i] = torch.maximum(1 - (1. - alpha) * lambd / (rho * torch.norm(s_td, p=2)+epsilon),
                                       torch.zeros_like(s_td)) * s_td
        return X

    def compute_Z(self, Z, R, X, rho):
        Z = Z + rho * (R - X)
        return Z



