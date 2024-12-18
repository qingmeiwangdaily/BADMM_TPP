import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import get_non_pad_mask
import matplotlib.pyplot as plt
import os
import random
import matplotlib
import seaborn as sns
import numpy as np


def softplus(x, beta):
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask):
    """ Log-likelihood of non-events, using linear interpolation. """

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device)
    temp_time /= (time[:, :-1] + 1).unsqueeze(2)

    temp_hid = model.linear(data)[:, 1:, :]
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


def log_likelihood(model, data, time, types):
    """ Log-likelihood of sequence. """

    non_pad_mask = get_non_pad_mask(types).squeeze(2)

    type_mask = torch.zeros([*types.size(), model.num_types], device=data.device)
    for i in range(model.num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(data.device)

    all_hid = model.linear(data)
    all_lambda = softplus(all_hid, model.beta)
    type_lambda = torch.sum(all_lambda * type_mask, dim=2)

    event_ll = compute_event(type_lambda, non_pad_mask)
    event_ll = torch.sum(event_ll, dim=-1)

    non_event_ll = compute_integral_unbiased(model, data, time, non_pad_mask, type_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll


def type_loss(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing. """

    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)

    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num


def time_loss(prediction, event_time):
    """ Time prediction loss. """

    prediction.squeeze_(-1)

    true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]

    diff = prediction - true
    se = torch.sum(diff * diff)
    return se


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss


def visualize_map(data: np.ndarray,
                  dst: str,
                 # ticklabels: List[str],
                  xlabel: str = None,
                  ylabel: str = None,
                  annot: bool = False,
                  vmin: float = 0,
                  vmax: float = 0.5,
                  cmap: str = "Reds",
                  rotation: int = None,
                  fontsize: int = 15):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'font.size': 20})
    # plt.figure(figsize=(int(1.3 * data.shape[0] + 1), int(1.3 * data.shape[1]) + 1))
    # plt.figure(figsize=(6, 5))
    ax = sns.heatmap(data,
                     linewidth=1,
                     vmin=vmin,
                     vmax=vmax,
                     square=True,
                     annot=annot,
                     cmap=cmap,
                     #xticklabels=ticklabels,
                     #yticklabels=ticklabels
                     #annot_kws={"fontsize": 15}
                     )
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)


    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)

    if rotation is not None:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, fontsize=fontsize)
    else:
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontdict={'size': fontsize})
    if xlabel is not None:
        plt.xlabel(xlabel, fontdict={'size': fontsize})
    plt.tight_layout()
    plt.savefig(dst)
    plt.close()