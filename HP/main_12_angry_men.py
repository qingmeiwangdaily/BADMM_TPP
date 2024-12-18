import numpy as np
from loss import loglike_loss,cal_accuracy,time_loss
import argparse
import pickle
from predict import predict
import time
import random
from EM import HawkesProcessEM
import torch
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def data_12_angry_men(opt):
    tree = ET.parse(opt.data)
    root = tree.getroot()

    num_types = 13

    result = []
    sentences = []
    for u in root.findall('.//u'):
        uid = int(u.attrib['uID'][1:])
        media_start = float(u.find('media').attrib['start'])
        result.append((media_start, uid))

        words = []
        for w in u.findall('w'):
            words.append(w.text)
        sentence = ' '.join(words)
        sentences.append((sentence,uid))

    result.sort()

    return result, num_types,sentences

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


def compute_rank(resp,sequence,num_types=13):
    col = np.sum(resp, axis=0)
    col_ = np.array([(value / m)  for m, value in zip(range(len(sequence), 0, -1), col)])
    
    events = np.flip(np.argsort(col_))  
    score = np.zeros(num_types)
    
    for m, (t_m, c_m) in enumerate(sequence):
        score[c_m] = score[c_m] + col[m]
    argrank = np.flip(np.argsort(score))
    rank = np.flip(np.sort(score))
    return rank,argrank,score,events

def seed_everything(opt):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    
def plot_and_save_matrix(matrix, filename):
    if len(matrix.shape) != 2:
        raise ValueError("The input matrix must be 2D.")
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap='inferno', vmin=0, vmax=1, origin='upper')
    fig.colorbar(cax)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(fig)

def train(model, train_data, opt):
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
       
        
        if epoch_i == opt.epoch - 1:
            with open(opt.result, 'a') as f:
                f.write('[Info] main parameters: A:{},mu:{}\n'.format(model.A,model.mu))
                f.write('[Info] all parameters: {}\n'.format(opt))
         

def main():
    """ Main function. """
    parser = argparse.ArgumentParser()
    parser.add_argument('-data',default= "../data/12-angry-men/12-angry-men.xml")
    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-log', type=str, default='log.txt')
    parser.add_argument('-seed', type=int, default=666)
    parser.add_argument('-alpha', type=float, default=0.1)
    parser.add_argument('-lambd', type=float, default=0.1)
    parser.add_argument('-rho', type=float, default=1.)
    parser.add_argument('-use_wandb', default=False)
    parser.add_argument('-result', type=str, default='result.txt')
    parser.add_argument('-mode', default = "BADMM_nuclear")
    parser.add_argument('-predict_time_nums', type=int, default = 2)
    parser.add_argument('-num_iteration', type=int, default =10)
    parser.add_argument('-project', type=str, default ='')
    parser.add_argument('-name', type=str, default ='')
    opt = parser.parse_args()
    
    seed_everything(opt)

    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """

   # train_data, valid_data,test_data, num_types = prepare_dataloader(opt)
    data,num_types,sentences = data_12_angry_men(opt)
    
    train_data = []
    
    train_data.append(data)

    model = HawkesProcessEM(num_types,opt,num_iterations=1)

    train(model,train_data,opt)
    
    rank,argrank,score,events = compute_rank(model.resp2[0],train_data[0],num_types=13)
    
            
    with open(opt.result, 'w') as f:
        f.write('[Info] main parameters: alpha:{},lambd:{}\n'.format(opt.alpha,opt.lambd))
        f.write("sorted score:\n")
        f.write(' '.join(map(str, rank)) + '\n') 
        f.write("sorted men:\n")
        f.write(' '.join(map(str, argrank)) + '\n')  
    
        for i in range(100):
            f.write("[Rank {} sentence]\n".format(i))
            f.write("  The man: {}\n  Sentence: {}\n".format(sentences[events[i]][1], sentences[events[i]][0]))

    print("sorted score:\n",rank,"sorted men:\n",argrank)

if __name__ == '__main__':
    main()



