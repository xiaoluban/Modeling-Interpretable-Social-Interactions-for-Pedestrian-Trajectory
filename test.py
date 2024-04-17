"""
test.py
Example: Testing on eth dataset
"""

import torch
import tqdm
import argparse
import os
import time
import numpy as np

from data_reconstruct import ST_GRAPH
from model import Interp_SocialLSTM
from utils_test import DataLoader


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
    parser.add_argument('--pretrained_model_index', type=int, default=None,
                        help='selected pretrained model for test')
    args = parser.parse_args()

    test(args)


def test(args):
    seed = 9
    torch.cuda.manual_seed(seed)
    seed0 = 4
    torch.manual_seed(seed0)
    seed1 = 0
    np.random.seed(seed1)
    torch.backends.deterministic = True
    torch.backends.benchmark = False

    _time = time.asctime(time.localtime(time.time()))

    data_dirs = ['../preprocessed data/eth/att-test']

    data_index = [0]
    datasets = [data_dirs[x] for x in data_index]
    datasets = [d for d in os.listdir(datasets[0])]

    dataloader = DataLoader(datasets, args.seq_length, args.pred_length, args.batch_size, data_dirs[0])
    stgraph = ST_GRAPH(args.batch_size, args.seq_length + args.pred_length)

    net = Interp_SocialLSTM(args, state='test')
    # print(net)

    checkpoint_path_trained = os.path.join('../save/basic/eth/k=' + str(args.k_head) + '/',
                                           'basic_lstm_model_' + str(args.pretrained_model_index) + '.tar')
    if os.path.isfile(checkpoint_path_trained):
        print('------Loading checkpoint---')
        checkpoint = torch.load(checkpoint_path_trained)
        model_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        print('-Loaded checkpoint at epoch-', model_epoch)

    test_epoch(dataloader, net, args, stgraph)


def test_epoch(dataloader, net, args, stgraph):
    dataloader.reset_batch_pointer()
    ade_total = 0.0
    fde_total = 0.0
    count_ade = 0
    count_fde = 0
    num_samples = 20

    for batch in tqdm.tqdm(range(dataloader.num_batches)):  # 81
        start = time.time()
        x, mask, _, _, _ = dataloader.next_batch()
        stgraph.readGraph(x)

        loss_batch = 0
        for sequence in range(8):
            nodes_temp, _, nodesPresent, _ = stgraph.getSequence(sequence)

            # with torch.no_grad():
            net.eval()
            _, ade, fde, _, _ = net(nodes_temp, nodesPresent, args, args.seq_length, args.pred_length, num_samples)
            ade_total += ade
            fde_total += fde

            if (fde - 0) < torch.exp(torch.Tensor([-6])):
                count_fde += 0
                count_ade += 0
            else:
                count_fde += 1
                count_ade += 1

        stgraph.reset()
        end = time.time()

    print('ade, fde', ade_total / count_ade, fde_total / count_fde)

    return True


if __name__ == '__main__':
    main()
