#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch.utils import data
from torchvision import datasets, transforms
from sampling import iid, noniid, mnist_noniid_unequal
import numpy as np
import time


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = noniid(train_dataset, args.num_users, args)

    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = noniid(train_dataset, args.num_users, args.user_max_class)
    elif args.dataset == 'emnist':
        data_dir = './data/emnist/'
        train = np.load(data_dir+"EMnist_train.npz", allow_pickle=True)
        test = np.load(data_dir+"EMnist_test.npz", allow_pickle=True)
        mean, std = train["stats"]
        x_train, y_train, user_idx_train = train["data"], train["label"], train["user_idx"]
        x_test, y_test, user_idx_test = test["data"], test["label"], test["user_idx"]

        y_train, y_test = y_train.astype(int), y_test.astype(int)
        x_train, x_test = x_train.reshape(-1, 1, 28, 28), x_test.reshape(-1, 1, 28, 28)
        x_train, x_test = x_train.astype(np.float)/255, x_test.astype(np.float)/255
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

        user_groups, threshold1, threshold2 = [], 100, 10
        for i in range(len(user_idx_train)):
            if len(user_idx_train[i]) >= threshold1 and len(user_idx_test[i]) >= threshold2:
                user_groups.append([user_idx_train[i], user_idx_test[i]])
        train_dataset = data.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
        test_dataset = data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return



def user_inference(idxs_users, global_model, LocalUpdate, args, train_dataset, user_groups, logger):
    user_train_accuracy_temp, user_train_loss_temp = [], []
    user_test_accuracy_temp, user_test_loss_temp = [], []
    user_size = []
    for idx in idxs_users:
        local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
        user_size.append(local_model.size)
        user_train_accuracy, user_train_loss = local_model.inference_trainloader(model=global_model)
        user_test_acc, user_test_loss = local_model.inference(model=global_model)
        user_train_accuracy_temp.append(user_train_accuracy)
        user_train_loss_temp.append(user_train_loss)
        user_test_accuracy_temp.append(user_test_acc)
        user_test_loss_temp.append(user_test_loss)
    return user_train_accuracy_temp, user_train_loss_temp, user_test_accuracy_temp, user_test_loss_temp, user_size
