import numpy as np
from loss import loglike_loss,cal_accuracy,time_loss
import argparse
import pickle
from predict import predict
import time
import random
from EM import HawkesProcessEM
import torch

def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """
    def convert_to_compact_format2(sequences):
        compact_format = []
        for seq in sequences:
            events = []
            for event in seq:
                time_event_tuple = (event['time_since_start'], event['type_event'])
                events.append(time_event_tuple)
            events.sort()  # Sort by time
            compact_format.append(events)
        return compact_format

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    train_data = convert_to_compact_format2(train_data)
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    dev_data = convert_to_compact_format2(dev_data)
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')
    test_data = convert_to_compact_format2(test_data)

    return train_data, dev_data, test_data, num_types


def seed_everything(opt):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

def train(model, train_data,valid_data,test_data, opt):
    """ Start training. """
    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmses = []  # validation event time prediction RMSE
    valid_rmses_norm = []
    test_event_losses = []  # test log-likelihood
    test_pred_losses = []  # test event type prediction accuracy
    test_rmses = []  # test event time prediction RMSE
    test_rmses_norm = []
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')
        #Training
        start = time.time()
        model.fit(train_data)
        pred_events = predict(model,train_data,opt.predict_time_nums)
        _,train_ll = loglike_loss(model,train_data)
        _,train_type_accuacy = cal_accuracy(train_data,pred_events)
        train_rmse_norm, train_rmse = time_loss(train_data,pred_events)

        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, RMSE_norm: {rmse_norm: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_ll, type=train_type_accuacy, rmse=train_rmse, rmse_norm=train_rmse_norm, elapse=(time.time() - start) / 60))

        #validating
        start = time.time()
        valid_events = valid_data
        pred_events = predict(model,valid_events,opt.predict_time_nums)
        _, valid_ll = loglike_loss(model, valid_events)
        _,valid_type_accuacy = cal_accuracy(valid_events, pred_events)
        valid_rmse_norm, valid_rmse = time_loss(valid_events, pred_events)

        print('  - (Validating)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f},  RMSE_norm: {rmse_norm: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_ll, type=valid_type_accuacy, rmse=valid_rmse, rmse_norm=valid_rmse_norm, elapse=(time.time() - start) / 60))

        #Testing
        start = time.time()
        test_events = test_data
        pred_events = predict(model, test_events,opt.predict_time_nums)
        _,test_ll = loglike_loss(model, test_events)
        _,test_type_accuacy = cal_accuracy(test_events, pred_events)
        test_rmse_norm, test_rmse = time_loss(test_events, pred_events)

        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f},  RMSE_norm: {rmse_norm: 8.5f},'
              'elapse: {elapse:3.3f} min'
              .format(ll=test_ll, type=test_type_accuacy, rmse=test_rmse, rmse_norm=test_rmse_norm, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_ll]
        valid_pred_losses += [valid_type_accuacy]
        valid_rmses += [valid_rmse]
        valid_rmses_norm += [valid_rmse_norm]
        test_event_losses += [test_ll]
        test_pred_losses += [test_type_accuacy]
        test_rmses += [test_rmse]
        test_rmses_norm += [test_rmse_norm]
        max_idx = np.argmax(valid_event_losses)
        print('  - [Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}, Minimum RMSE_norn: {rmse_norm: 8.5f}'
              .format(event=test_event_losses[max_idx], pred=test_pred_losses[max_idx], rmse=test_rmses[max_idx], rmse_norm=test_rmses_norm[max_idx]))

        # logging
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}, {rmse_norm: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_ll, acc=valid_type_accuacy, rmse=valid_rmse, rmse_norm=valid_rmse_norm))
    
        if epoch_i == opt.epoch - 1:
            checkpoint={
                'epoch': epoch,
                'num_types': model.num_types,
                'num_iterations': model.num_iterations,
                'mu': model.mu,
                'A': model.A,
                'opt': model.opt,
                'resp1': model.resp1,
                'resp2': model.resp2
            }
            with open(opt.checkpoint, 'wb') as f:
                pickle.dump(checkpoint, f)

        if epoch_i > 0:
            with open(opt.result, 'a') as f:
                f.write('[Info] all parameters: {}\n'.format(opt))
                f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}, {rmse_norm: 8.5f}\n'
                        .format(epoch=max_idx, ll=test_event_losses[max_idx], acc=test_pred_losses[max_idx],
                                rmse=test_rmses[max_idx], rmse_norm=test_rmses_norm[max_idx]))


def main():
    """ Main function. """
    parser = argparse.ArgumentParser()
    parser.add_argument('-data',default= "../data/retweet/")
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-log', type=str, default='log.txt')
    parser.add_argument('-seed', type=int, default=666)
    parser.add_argument('-lambd', type=float, default=0.6)
    parser.add_argument('-alpha', type=float, default=0.02)
    parser.add_argument('-rho', type=float, default=1.)
    parser.add_argument('-use_wandb', default=False)
    parser.add_argument('-result', type=str, default='result.txt')
    parser.add_argument('-predict_time_nums', type=int, default = 2)
    parser.add_argument('-checkpoint', type=str, default ='')
    parser.add_argument('-num_iteration', type=int, default =10)
    parser.add_argument('-pre_model', default=False)
    parser.add_argument('-mode',type=str, default = "BADMM_nuclear")# choose module:BADMM_nuclear/BADMM12/EM

    opt = parser.parse_args()
    seed_everything(opt)

    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """

    train_data, valid_data,test_data, num_types = prepare_dataloader(opt)

    model = HawkesProcessEM(num_types, opt, num_iterations=1)  # use opt.model to choose model
    
    if opt.pre_model == True:
       # print(opt.pre_model)
        with open(opt.checkpoint,'rb') as f:
            checkpoint = pickle.load(f)
        model.num_types = checkpoint['num_types']
        model.num_iterations = checkpoint['num_iterations']
        model.mu = checkpoint['mu']
        model.A = checkpoint['A']
        model.opt = checkpoint['opt']
        model.resp1 = checkpoint['resp1']
        model.resp2 = checkpoint['resp2']
        print("load model success......")

    train(model, train_data, valid_data, test_data, opt)

if __name__ == '__main__':
    main()



