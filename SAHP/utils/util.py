'''
  Utility functions for CTLSTM model
'''

import torch
from torch import nn
from utils.load_synth_data import one_hot_embedding
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib
from typing import List, Union

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
                  fontsize: int = 8):
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
    cbar.ax.tick_params(labelsize=10)

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

def plot_and_save_matrices(matrices, filename):
    n = len(matrices)

    fig, axs = plt.subplots(1, n, figsize=(4*n, n//2))
    # fig, axs = plt.subplots(2, n//2, figsize=(2*n, n))

    for i in range(n):
        ax = axs[i]
        matrix = matrices[i]
        cax = ax.imshow(matrix, cmap='inferno', vmin=0, vmax=1, origin='upper')
        fig.colorbar(cax, ax=ax)
        ax.set_title(f"attention {i}")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)

def seed_everything(seed=666):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_batch(batch_size, i_batch, model, seq_lengths, seq_times, seq_types, rnn = True):
    start_pos = i_batch
    end_pos = i_batch + batch_size
    batch_seq_lengths = seq_lengths[start_pos:end_pos]
    max_seq_length = batch_seq_lengths[0]
    batch_seq_times = seq_times[start_pos:end_pos, :max_seq_length + 1]
    batch_seq_types = seq_types[start_pos:end_pos, :max_seq_length + 1]
    # Inter-event time intervals
    batch_dt = batch_seq_times[:, 1:] - batch_seq_times[:, :-1]

    batch_onehot = one_hot_embedding(batch_seq_types, model.input_size)
    batch_onehot = batch_onehot[:, :, :model.process_dim]# [1,0], [0,1], [0,0]

    if rnn:
        # Pack the sequences for rnn
        packed_dt = nn.utils.rnn.pack_padded_sequence(batch_dt, batch_seq_lengths, batch_first=True)
        packed_types = nn.utils.rnn.pack_padded_sequence(batch_seq_types, batch_seq_lengths, batch_first=True)
        max_pack_batch_size = packed_dt.batch_sizes[0]
    else:
        # self-attention
        packed_dt,packed_types,max_pack_batch_size = None, None,0
    return batch_onehot, batch_seq_times, batch_dt, batch_seq_types, \
           max_pack_batch_size, packed_dt, packed_types, batch_seq_lengths

def generate_sim_interval_seqs(interval_seqs, seqs_length):
    ''' Generate a simulated time interval sequences from original time interval sequences
        based on uniform distribution

    Args:
        interval_seqs: list of torch float tensors
    Results:
        sim_interval_seqs: list of torch float tensors
        sim_index_seqs: list of torch long tensors
    '''
    sim_interval_seqs = torch.zeros((interval_seqs.size()[0], interval_seqs.size()[1]-1)).float()
    sim_index_seqs = torch.zeros((interval_seqs.size()[0], interval_seqs.size()[1]-1)).long()
    restore_interval_seqs, restore_sim_interval_seqs = [], []
    for idx, interval_seq in enumerate(interval_seqs):
        restore_interval_seq = torch.stack([torch.sum(interval_seq[0:i]) for i in range(1,seqs_length[idx]+1)])
        # Generate N-1 time points. Here N includes <BOS>
        restore_sim_interval_seq, _ = torch.sort(torch.empty(seqs_length[idx]-1).uniform_(0, restore_interval_seq[-1]))

        sim_interval_seq = torch.zeros(seqs_length[idx]-1)
        sim_index_seq = torch.zeros(seqs_length[idx]-1).long()

        for idx_t, t in enumerate(restore_interval_seq):
            indices_to_update = restore_sim_interval_seq > t

            sim_interval_seq[indices_to_update] = restore_sim_interval_seq[indices_to_update] - t
            sim_index_seq[indices_to_update] = idx_t

        restore_interval_seqs.append(restore_interval_seq)
        restore_sim_interval_seqs.append(restore_sim_interval_seq)
        sim_interval_seqs[idx, :seqs_length[idx]-1] = sim_interval_seq
        sim_index_seqs[idx, :seqs_length[idx]-1] = sim_index_seq

    return sim_interval_seqs

def pad_bos(batch_data, type_size):
    event_seqs, interval_seqs, total_interval_seqs, seqs_length = batch_data
    pad_event_seqs = torch.zeros((event_seqs.size()[0], event_seqs.size()[1]+1)).long() * type_size
    pad_interval_seqs = torch.zeros((interval_seqs.size()[0], event_seqs.size()[1]+1)).float()

    pad_event_seqs[:, 1:] = event_seqs.clone()
    pad_event_seqs[:, 0] = type_size
    pad_interval_seqs[:, 1:] = interval_seqs.clone()

    return pad_event_seqs, pad_interval_seqs, total_interval_seqs, seqs_length


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    a = torch.tensor([0., 1., 2., 3., 4., 5.])
    b = torch.tensor([0., 2., 4., 6., 0., 0.])
    sim_interval_seqs, sim_index_seqs = generate_sim_interval_seqs(torch.stack([a,b]), torch.LongTensor([6,4]))
    print(sim_interval_seqs)
    print(sim_index_seqs)

