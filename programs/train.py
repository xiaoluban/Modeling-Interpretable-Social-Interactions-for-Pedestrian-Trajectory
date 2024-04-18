"""
train.py
Example: Testing on eth dataset
"""

import argparse
import os
import pickle
import time
import numpy as np
import torch
import tqdm

from data_reconstruct import ST_GRAPH
from model import Interp_SocialLSTM
from utils_train import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_head', type=int, help='K-the number of the interaction modes')
    # input size of location (x,y, offset x, offset y)
    parser.add_argument('--input_size', type=int, default=4)
    # output of guassian variables() or (x, y)
    parser.add_argument('--output_size', type=int, default=2)
    parser.add_argument('--rnn_size', type=int, default=128)
    # size of each batch, bach_size is the number of the sequence .
    # each sequence with length of observation length + prediction length
    parser.add_argument('--batch_size', type=int, default=8,
                        help='minibatch size')
    # observation sequence length
    parser.add_argument('--seq_length', type=int, default=8,
                        help='observation length')
    # prediction sequence length
    parser.add_argument('--pred_length', type=int, default=12,
                        help='prediction length')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for RMSProp')
    parser.add_argument('--lambda_rate', type=float, default=0.95,
                        help='weight_decay of adagrad')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='dropout probability')
    parser.add_argument('--grad_clip', type=float, default=10,
                        help='clip gradients at this time')
    parser.add_argument('--embedding_size', type=int, default=128,
                        help='embedding dimension of input data')
    parser.add_argument('--neighbor_size', type=int, default=4,
                        help='neighborhood size')
    parser.add_argument('--num_gaussians', type=float, default=5,
                        help='number of gaussians')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='L2 regularization parameter')
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use GPU or not')
    args = parser.parse_args()

    train(args)


def train(args):
    seed = 8
    torch.cuda.manual_seed(seed)
    seed0 = 8
    torch.manual_seed(seed0)
    seed1 = 8
    np.random.seed(seed1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _time = time.asctime(time.localtime(time.time()))

    data_dirs = ['../preprocessed data/eth/att-train/']
    scaler_train = 0  # get_scaler('train')
    val_data_dirs = ['../preprocessed data/eth/att-validation/']
    scaler_val = 0  # get_scaler('val')

    data_index = [0]
    datasets = [data_dirs[x] for x in data_index]
    datasets = [d for d in os.listdir(datasets[0])]

    dataloader = DataLoader(datasets, args.seq_length,
                            args.pred_length, args.batch_size, data_dirs[0], True)
    stgraph = ST_GRAPH(args.batch_size, args.seq_length + args.pred_length)

    val_datasets = [val_data_dirs[x] for x in data_index]
    val_datasets = [d for d in os.listdir(val_datasets[0])]
    val_dataloader = DataLoader(val_datasets, args.seq_length,
                                args.pred_length, args.batch_size, val_data_dirs[0], False)
    val_stgraph = ST_GRAPH(args.batch_size, args.seq_length + args.pred_length)

    log_directory = '../log/basic/eth/k=' + str(args.k_head) + '/'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_file_curve = open(os.path.join(log_directory, 'log_curve.txt'), 'w+')
    log_val_curve = open(os.path.join(
        log_directory, 'log_val_curve.txt'), 'w+')

    save_directory = '../save/basic/eth/k=' + str(args.k_head) + '/'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    with open(os.path.join(save_directory, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    def checkpoint_path(x):
        return os.path.join(save_directory, 'basic_lstm_model_' + str(x) + '.tar')

    net = Interp_SocialLSTM(args, state='train')
    print(net)
    if args.use_cuda:
        net = net.cuda()

    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.learning_rate, eps=1e-3, amsgrad=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # initiate early stop
    earlystop = False
    bestscore = None
    thres_patience = 80
    count_patience = 0
    print('training begin')

    # training begin
    for epoch in range(args.num_epochs):
        with torch.enable_grad():
            dataloader, stgraph, net, optimizer = train_epoch(dataloader, net, args, stgraph, epoch, optimizer,
                                                              log_file_curve)
            val_loss_batch, _ = validation(
                val_dataloader, net, args, val_stgraph, epoch, log_val_curve)
        scheduler.step(val_loss_batch)

        score = val_loss_batch
        if bestscore is None:
            bestscore = score
            print('saving model')
            torch.save({'epoch': epoch,
                        'state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, checkpoint_path(epoch))
        elif score > bestscore:
            count_patience += 1
            print('EarlyStopping counter: {} out of {}'.format(
                count_patience, thres_patience))
            if count_patience >= thres_patience:
                earlystop = True
        else:
            bestscore = score
            print('saving model')
            torch.save({'epoch': epoch,
                        'state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, checkpoint_path(epoch))

            count_patience = 0

        if earlystop is True:
            print('training End')
            break

    log_file_curve.close()
    log_val_curve.close()


def validation(dataloader, net, args, stgraph, epoch, log_file_curve):
    dataloader.reset_batch_pointer()
    loss_epoch = 0.0
    for batch in range(dataloader.num_batches):  # 81
        start = time.time()
        x, mask, _, _ = dataloader.next_batch()
        stgraph.readGraph(x)

        loss_batch = 0

        for sequence in range(args.batch_size):
            nodes_temp, _, nodesPresent, _ = stgraph.getSequence(sequence)

            loss, _, _, _ = net.run_train(nodes_temp, nodesPresent,
                                          args, args.seq_length, args.pred_length)

            if loss == 0:
                continue

            loss_batch += loss.item()

        stgraph.reset()
        end = time.time()

        loss_batch = loss_batch / 8
        loss_epoch += loss_batch

    loss_epoch = loss_epoch / dataloader.num_batches

    print(
        'epoch{}(batch num {}), valid_loss = {:.3f}, time = {:.3f}'.format(
            epoch,
            dataloader.num_batches,
            loss_epoch,
            end - start))
    log_file_curve.write(str(epoch) + ',' + str(loss_epoch) + ',')

    return loss_epoch, _


def train_epoch(dataloader, net, args, stgraph, epoch, optimizer, log_file_curve):
    """
    we need to generate the whole observation for one or two agents
    """

    dataloader.reset_batch_pointer()
    loss_epoch = 0.0

    for batch in range(dataloader.num_batches):  # 81
        start = time.time()
        x, mask, _, _ = dataloader.next_batch()
        stgraph.readGraph(x)

        loss_batch = 0

        for sequence in range(args.batch_size):
            nodes_temp, _, nodesPresent, _ = stgraph.getSequence(sequence)

            loss, _, _, _ = net.run_train(nodes_temp, nodesPresent,
                                          args, args.seq_length, args.pred_length)

            if loss == 0:
                continue

            loss_batch += loss.item()

            optimizer.zero_grad()

            loss.backward()

            # clip gradient of lstm
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()

        stgraph.reset()
        end = time.time()

        loss_batch = loss_batch / 8
        loss_epoch += loss_batch

    loss_epoch = loss_epoch / dataloader.num_batches

    print(
        'epoch{}(batch num {}), train_loss = {:.3f}, time = {:.3f}'.format(
            epoch,
            dataloader.num_batches,
            loss_epoch,
            end - start))
    log_file_curve.write(str(epoch) + ',' + str(loss_epoch) + ',')

    return dataloader, stgraph, net, optimizer


if __name__ == '__main__':
    main()
