#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inference', type=int, default=0,
                        help="Do inference during training for recording more advance information")

    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=20,
                        help="local batch size: B")
    parser.add_argument('--full_batch', type=int, default=0,
                        help="whether to use mini-batch gradient or full-batch gradient descent;"
                             " it overrules the --local_bs option")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_serv', type=float, default=1,
                        help='server side learning rate')
    parser.add_argument('--decay_rate', type=float, default=1,
                        help='the decay rate for the server side learning rate; {<=1 or 2 for adaptive}')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")
    # MGDA arguments
    parser.add_argument('--normalize', type=int, default=0,
                        help="Default set to no normalization. Set to 1 for normalization")
    parser.add_argument('--epsilon', type=float, default=1.,
                        help="Interpolation between FedMGDA and FedAvg. \
                        When set to 0, recovers FedAvg; When set to 1, is FedMGDA without any constraint")
    # VIP
    parser.add_argument('--vip', type=int, default=-1,
                        help='the ID of a user that participates in each communication round; {-1 no vip, 0....number of users}')
    parser.add_argument('--vip_scale', type=float, default=1,
                        help='scales the loss function of the vip user by a constant number')
    parser.add_argument('--vip_bias', type=float, default=0,
                        help='adds a constant to  the loss function of the vip user ')

    # Proximal arguments
    parser.add_argument('--prox_weight', type=float, default=0.0, help='the weight of proximal regularization term in FedProx and FedMGDA')

    # Q-fair federated learning
    parser.add_argument("--qffl", type=float, default=0.0, help="the q-value in the qffl algorithm. \
                                                                qffl with q=0 reduces to FedAvg")
    parser.add_argument('--Lipschitz_constant', type=float, default=1.0)

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--user_max_class', type=int, default=5,
                        help='Maximum number of data classes per use for non-IID.')

    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument("--save_dir", type=str, help="Path to directory in which to dump output")

    # Only for splitting algorithms
    parser.add_argument('--method', type=str, default='fedavg', help="fl algorithm: {fedavg, fedDR, FedPR, FedInv}")

    # Anderson Accelerated
    parser.add_argument('--memory', type=int, default=1, help="length of the weight memory for Anderson acceleration")
    parser.add_argument('--acceleration', type=int, default=0, help="whether to use acceleration. "
                                                                    "Set to 1 to activate the feature")
    parser.add_argument("--acc_avg", type=float, default=0.0, help="weight of averaging between acceleration and normal step")


    # initializing from a a saved model
    parser.add_argument('--reuse', type=str, default='False',
                        help="initializing from a a saved model")

    args = parser.parse_args()
    return args


