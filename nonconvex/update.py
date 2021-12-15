#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.nn.utils.convert_parameters import vector_to_parameters
import numpy as np
import time


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        # self.full_batch_size = len(idxs[0])
        self.trainloader, self.validloader, self.testloader, self.size = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        if len(idxs) == 2:
            idxs_train = list(idxs[0])
            idxs_val = []
            idxs_test = list(idxs[1])
        else:
            idxs_train = idxs[:int(0.8*len(idxs))]
            idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
            idxs_test = idxs[int(0.9*len(idxs)):]

        if self.args.full_batch == 1:
            trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                     batch_size=len(idxs_train), shuffle=True)
            # print("full_batch: {}".format(len(idxs_train)))
        else:
            trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                     batch_size=self.args.local_bs, shuffle=True)
        if idxs_val:
            validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                     batch_size=int(len(idxs_val)/5), shuffle=False)
        else:
            validloader = []
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=len(idxs_test), shuffle=False)
        return trainloader, validloader, testloader, [len(idxs_train), len(idxs_val), len(idxs_test)]

    def update_weights(self, model, global_round, user_index):
        # Set mode to train model

        ######################
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        # if user_index == self.args.vip:
        #     loss = self.args.vip_scale * loss + self.args.vip_bias
        accuracy = correct / total
        # print("before", accuracy)
        ######################

        model.train()
        epoch_loss = []

        old_weights = copy.deepcopy(model.state_dict())
        list_param = list(model.parameters())
        weight_regularizer = nn.MSELoss(reduction='sum')

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()

                log_probs = model(images)

                if self.args.prox_weight:
                    net_reg = [0]
                    for i, j in enumerate(old_weights):
                        net_reg.append(net_reg[-1] + weight_regularizer(list_param[i], old_weights[j]))
                    if self.args.decay_rate == 1:
                        prox_weight = self.args.prox_weight
                    elif self.args.decay_rate == 2:
                        prox_weight = self.args.prox_weight / (global_round+1)
                    elif self.args.decay_rate == 3:
                        prox_weight = self.args.prox_weight / np.sqrt(global_round+1)
                    elif self.args.decay_rate == 4:
                        prox_weight = self.args.prox_weight / np.log(global_round+np.exp(1))
                    elif self.args.decay_rate == 5:
                        prox_weight = self.args.prox_weight / 2**int(global_round/10)
                    loss = self.criterion(log_probs, labels) + (1/(2*prox_weight)) * net_reg[-1]
                else:
                    loss = self.criterion(log_probs, labels)

                if user_index == self.args.vip:
                    loss = self.args.vip_scale * loss
                # loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        ######################
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        # if user_index == self.args.vip:
        #     loss = self.args.vip_scale * loss + self.args.vip_bias
        accuracy = correct/total
        # print("after", accuracy)
        ######################

        difference = copy.deepcopy(old_weights)

        with torch.no_grad():
            for key in difference.keys():
                difference[key] = model.state_dict()[key] - old_weights[key]

        # normalize the gradient
        # s = time.time()
        total = []
        if self.args.normalize == 1:
            total = 0.0  # the norm to compute
            for key in difference.keys():
                total += torch.norm(difference[key])**2
            total = np.sqrt(total.item())
            for key in difference.keys():
                difference[key] /= total
        # print(user_index, total)

        # return difference, sum(epoch_loss) / len(epoch_loss), loss, accuracy, total
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), loss, accuracy, total

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

    def inference_trainloader(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
