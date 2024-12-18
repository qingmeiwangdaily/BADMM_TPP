import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader
from transformer.Models import Transformer
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import os
import random
import matplotlib
import seaborn as sns

torch.autograd.set_detect_anomaly(True)

def seed_everything(seed=666):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """
    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=False)
    validationloader = get_dataloader(dev_data, opt.batch_size, shuffle=False)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, validationloader, testloader, num_types


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()

        enc_out, prediction, slf_attn = model(event_type, event_time)

        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = Utils.time_loss(prediction[1], event_time)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 100
        loss = event_loss + pred_loss + se / scale_time_loss
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse, slf_attn


def eval_epoch(model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

            """ forward """
            enc_out, prediction, slf_attn = model(event_type, event_time)

            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)
            _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            se = Utils.time_loss(prediction[1], event_time)

            """ note keeping """
            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse, slf_attn


def train(model, training_data, validation_data, test_data, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    test_event_losses = []  # test log-likelihood---
    test_pred_losses = []  # test event type prediction accuracy---
    test_rmse = []  # test event time prediction RMSE---

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time, slf_attn_train = train_epoch(model, training_data, optimizer, pred_loss_func, opt)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event, valid_type, valid_time, slf_attn_val = eval_epoch(model, validation_data, pred_loss_func, opt)
        print('  - (Validating)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

        start = time.time()
        test_event, test_type, test_time, slf_attn_test = eval_epoch(model, test_data, pred_loss_func, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=test_event, type=test_type, rmse=test_time, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        test_event_losses += [test_event]
        test_pred_losses += [test_type]
        test_rmse += [test_time]
        max_idx = np.argmax(valid_event_losses)  
        print('  - [Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
              .format(event=test_event_losses[max_idx], pred=test_pred_losses[max_idx], rmse=test_rmse[max_idx]))


        with open(opt.log, 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch, ll=test_event_losses[max_idx], acc=test_pred_losses[max_idx], rmse=test_rmse[max_idx]))

        if opt.use_wandb:
            wandb.log({
                'epoch': epoch,
                'val_ll': valid_event,
                'val_acc': valid_type,
                'val_rmse': valid_time,
                'test_ll': test_event,
                'test_acc': test_type,
                'test_rmse': test_time,
                'Max_ll': test_event_losses[max_idx],
                'Max_acc': test_pred_losses[max_idx],
                'Min_rmse': test_rmse[max_idx]
            })
        if test_type >= test_pred_losses[max_idx]:
            torch.save(model, f'{opt.checkpoints_path}/{opt.wandb}_{opt.name}_{opt.seed}_bestacc.pth')
            print(f'**************** save {epoch + 1}-th trained model as bestacc_checkpoint ****************')
        if(epoch == opt.epoch-1):
            torch.save(model, f'{opt.checkpoints_path}/{opt.wandb}_{opt.name}_{opt.seed}_best.pth')
            print(f'**************** save {epoch + 1}-th trained model as best_checkpoint ****************')

        scheduler.step()


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4) 
    parser.add_argument('-n_layers', type=int, default=4) 

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)

    parser.add_argument('-mode', type=str, default='softmax')  #badmm/badmm12/sinkhorn
    parser.add_argument('-log', type=str, default='')
    parser.add_argument('-use_wandb', type=bool, default=False)
    parser.add_argument('-n_it', type=int, default=2)
    parser.add_argument('-rho', type=float, default=1)
    parser.add_argument('-lambda_', type=float, default=0.1)
    parser.add_argument('-alpha', type=float, default=0.1)
    parser.add_argument('-seed', type=int, default=666)
    parser.add_argument('-wandb', type=str, default='')
    parser.add_argument('-name',type=str,default='')
    parser.add_argument('-checkpoints_path', type=str, default='./checkpoints')

    opt = parser.parse_args()

    opt.device = torch.device('cuda')
    seed_everything(opt.seed)  # seed

    if opt.use_wandb:
        wandb.login(key="")
        wandb.init(project=opt.wandb,name=opt.name)

    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    trainloader, validationloader, testloader, num_types = prepare_dataloader(opt)

    """ prepare model """
    # model =torch.load(opt.checkpoints_path)
    # print(f'**************** load model from {opt.checkpoints_path} ****************')
    model = Transformer(
        num_types=num_types,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
        n_it=opt.n_it,
        mode=opt.mode,
        rho=opt.rho,
        lambda_=opt.lambda_,
        alpha=opt.alpha
    )
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1) 
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none') 

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    train(model, trainloader, validationloader, testloader, optimizer, scheduler, pred_loss_func, opt)


if __name__ == '__main__':
    main()
