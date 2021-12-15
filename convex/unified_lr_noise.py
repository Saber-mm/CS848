import numpy as np
import torch
import copy
import argparse
import torch as torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import math as math
import pickle

def main(x0, A, b, comm_rounds, eta=0.01, proximal_update_epochs=100, proximal_update_lr=0.01, fedAVG_local_epochs=2, fedAVG_lr=0.0001):
    

    x_orig = copy.deepcopy(x0)
    print('finding the global optimizer using GD for sigma_noise {} ...'.format(sigma_noise))
    x_opt, obj_opt = minimiser(A, b, lr=0.0001, epochs=40000)
    heterogeneity = heterogeneity_measure(x_opt, A, b)

    np.random.seed(100)
    torch.manual_seed(100)
    
    # setting parameters based on the chosen method:
    if args.method == 'fedAVG' or args.method == 'fedProx':
        alpha = 1.0
        beta = 1.0
        gamma = 1.0
    elif args.method == 'fedSplit':
        alpha = 2.0
        beta = 2.0
        gamma = 1.0
    elif args.method == 'fedPI':
        alpha = 2.0
        beta = 2.0
        gamma = 0.5
    elif args.method == 'fedRP':
        alpha = 2.0
        beta = 1.0
        gamma = 1.0

    # initialization:
    model_global = LogisticRegression(A[0].shape[1])
    x_central = model_global.state_dict()['linear.weight'].numpy()
    z_clients = [x_central for c in range(len(A))]
    w_clients = [x_central for c in range(len(A))]
    u_clients = [x_central for c in range(len(A))]
    
    obj_val = []
    ###########################
    if args.method != 'fedAVG':
        for e in range(comm_rounds):

            if e%10==0:
                print("at communication round {}".format(e))

            for j in range(len(A)):
                z_clients[j] = ((1-alpha)*u_clients[j]) + (alpha*prox_update(copy.deepcopy(u_clients[j]), copy.deepcopy(model_global), 
                A[j], b[j], eta, proximal_update_epochs, proximal_update_lr))
            
            # projecting local solutions on the equality subspace:
            x_central = model_avg(z_clients)
            global_model_parameters_dict = model_global.state_dict()
            for k in global_model_parameters_dict.keys():
                if k == 'linear.weight':
                    global_model_parameters_dict[k] = torch.tensor(x_central)
            model_global.load_state_dict(global_model_parameters_dict)        
            
            obj_val.append(objective(copy.deepcopy(model_global), A, b).numpy())
            
            # finding the next iteration proximal point for each user:
            for j in range(len(A)):
                w_clients[j] = ((1-beta)*z_clients[j]) + (beta*x_central)
                u_clients[j] = ((1-gamma)*u_clients[j]) + (gamma*w_clients[j])      
    
    #############################
    elif args.method == 'fedAVG':
        for e in range(comm_rounds):

            if e%10==0:
                print("at communication round {}".format(e))
            
            for j in range(len(A)):
                z_clients[j] = ((1-alpha)*u_clients[j]) + (alpha*fedAVG_model_update(copy.deepcopy(model_global), A[j], b[j], fedAVG_local_epochs, fedAVG_lr))
            
            # projecting local solutions on the equality subspace:
            x_central = model_avg(z_clients)
            global_model_parameters_dict = model_global.state_dict()
            for k in global_model_parameters_dict.keys():
                if k == 'linear.weight':
                    global_model_parameters_dict[k] = torch.tensor(x_central)
            model_global.load_state_dict(global_model_parameters_dict)        
            
            obj_val.append(objective(copy.deepcopy(model_global), A, b).numpy())
            
            # finding the next iteration proximal point for each user:
            for j in range(len(A)):
                w_clients[j] = ((1-beta)*z_clients[j]) + (beta*x_central)
                u_clients[j] = ((1-gamma)*u_clients[j]) + (gamma*w_clients[j])    
    ############################
    
    return np.array(obj_val), obj_opt, heterogeneity


def objective(model, A, b):
    m = len(A)
    model.eval()
    with torch.no_grad():
        obj = 0
        for i in range(m):
            obj += criterion(b[i], A[i], model)
    return obj / m



def heterogeneity_measure(x_opt, A, b):
    n, d = A[0].shape
    m = len(A)
    model = LogisticRegression(d)

    ### loading the model with the global optimizer:
    model_params_dict = model.state_dict()
    for k in model_params_dict.keys():
            if k == 'linear.weight':
                model_params_dict[k] = x_opt['linear.weight']
    model.load_state_dict(model_params_dict)
    ###

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    model.train()
    sum_norm_2 = 0
    for i in range(m):
        optimizer.zero_grad()
        loss = criterion(b[i], A[i], model) # computing the global "total" loss
        loss.backward() # backpropagating over "total" loss
        sum_norm_2 += torch.norm(model.linear.weight.grad)**2
    
    return sum_norm_2/m




def prox_update(prox_point_j, model, aj, bj, eta, proximal_update_epochs, proximal_update_lr):
    
    optimizer = torch.optim.SGD(model.parameters(), lr=proximal_update_lr)
    proximal_point_j = torch.from_numpy(prox_point_j)
    
    model.train()
    for t in range(proximal_update_epochs):

        model.zero_grad()

        fj = criterion(bj, aj, model)
        if eta == np.inf:
            proximalmap_objective = fj.double()
        elif eta != np.inf:
            model_parameters = torch.transpose(model.linear.weight, 0, 1)
            proximal_regularizer = (torch.norm(proximal_point_j.double()-torch.transpose(model_parameters.double(),0,1))**2)/(2*eta)
            proximalmap_objective = fj.double() + proximal_regularizer

        proximalmap_objective.backward()
        optimizer.step()

    return  model.state_dict()['linear.weight'].numpy()

def fedAVG_model_update(model, A, b, l_ep, lr):

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    model.train()
    for t in range(l_ep):
        model.zero_grad()
        loss = criterion(b, A, model)
        loss.backward()
        optimizer.step()

    return model.state_dict()['linear.weight'].numpy()


def model_avg(models):
    x = np.zeros(models[0].shape)
    for i in range(len(models)):
        x += models[i]
    return x/len(models)


def minimiser(A, b, lr, epochs):
    n, d = A[0].shape
    m = len(A)
    model = LogisticRegression(d)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    model.train()
    for t in range(epochs):
        optimizer.zero_grad()
        loss = 0
        for i in range(m):
            loss += criterion(b[i], A[i], model) # computing the global "total" loss
        loss.backward() # backpropagating over "total" loss
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        obj1 = 0
        for i in range(m):
            obj1 += criterion(b[i], A[i], model)
    x = model.state_dict()
    return x, obj1 / m


def criterion(b, a, model):
    model_parameters = torch.transpose(model.linear.weight, 0, 1)
    p = torch.mul(b, torch.matmul(a, model_parameters))
    neglikelihood = torch.sum(torch.log(1/torch.sigmoid(p)))
    regularizer = (torch.norm(model_parameters)**2)/(2*m*n)
    return neglikelihood + regularizer

def data_gen(sigma_noise, m=10, d=100, n=1000, sigma_a=1):
    np.random.seed(100)
    torch.manual_seed(100)
    
    x0 = torch.randn(d).reshape(-1, 1)
    A = []
    b = []
    
    for j in range(m):
        a = torch.randn(n*d).reshape(n, d)
        A.append(a)

    percent = 0
    for j in range(m):

        dis = torch.matmul(A[j], x0)
        prob = torch.sigmoid(dis)# - 0.5

        samples = sigma_noise*torch.randn(n,1)
        labels_flip_mask = samples
        for k in range(len(samples)):
            if torch.abs(samples[k])<0.5:
                labels_flip_mask[k]=0.0
            else:
                labels_flip_mask[k]=1.0 
        percent += (torch.sum(labels_flip_mask)/len(labels_flip_mask))/m

        labels = (2*torch.bernoulli(prob))-1
        b.append(torch.mul(1-labels_flip_mask,labels)+torch.mul(labels_flip_mask,-labels))
        #b.append(labels)
    print('on average {}% of the labels were flipped'.format(percent*100))
    return x0, A, b


class LogisticRegression(torch.nn.Module):
    def __init__(self, d):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(d, 1, bias=False)

    def forward(self, x):
        prob = torch.sigmoid(self.linear(x))
        return prob


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--comm_rounds', type=int, default=805, help="number of rounds of training")
    parser.add_argument('--method', type=str, default='fedSplit', help="method in {fedAVG, fedProx, fedSplit, fedPI, fedRP}")
    parser.add_argument('--eta_stepsize', type=float, default=0.01, help='proximal mapping step size (recommended value for FedProx: 0.001, for rest: 0.01)')
    ###
    parser.add_argument('--prox_lep', type=int, default=100, help='epcohs for proximal operator')
    parser.add_argument('--prox_lr', type=float, default=0.01, help='learning rate for pSroximal operator (recommended value around: 0.001 (the smallet the eta_stepsize the smaller prox_lr))')
    ###
    parser.add_argument('--fedAVG_lep', type=int, default=2, help='local epochs for fedAVG')
    parser.add_argument('--fedAVG_lr', type=float, default=0.01, help='local learning rate for fedAVG')
    parser.add_argument('--option', type=int, default=3, help="option in {1, 2, 3, 22} for fedAVG")
    parser.add_argument('--seed', type=int, default=250, help='random seed')
    m, n, d = 10, 1000, 100  # number of users, number of points of each user, dimensionality 
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    noise = [0, 0.257]
    ###########################
    if args.method != 'fedAVG':
        Eta = args.eta_stepsize * np.array([1])
        idx = [1]

        labels = []
        obj_vals = []
        heterogeneity_values = []

        for eta in Eta:
            for sigma_noise in noise:
                print("eta: {}".format(eta))
                print("sigma noise: {}".format(sigma_noise))
                x0, A, b = data_gen(sigma_noise)
                temp, obj_opt, heterogeneity = main(x0, A, b, comm_rounds=args.comm_rounds, eta=eta, proximal_update_epochs=args.prox_lep, proximal_update_lr=args.prox_lr)
                heterogeneity_values.append(heterogeneity)
                obj_vals.append(temp - obj_opt.numpy())

        j=0
        for sigma_noise in noise:
            labels.append(r"$\eta = 10^{{{}}}, heterogeneity\approx{:.3f} \times 10^{{{}}}$".format(np.log10(Eta[0]).astype(int), heterogeneity_values[j]/1000, "3"))
            j+=1        


        
    #############################
    elif args.method == 'fedAVG':
        local_epoch = [args.fedAVG_lep]
        learning_rate = [args.fedAVG_lr] * len(local_epoch)
        labels = []
        
        
        obj_vals = []
        heterogeneity_values = []

        for lep, lr in zip(local_epoch, learning_rate):
            print(lep, lr)
            for sigma_noise in noise:
                print("sigma noise: {}".format(sigma_noise))
                x0, A, b = data_gen(sigma_noise=sigma_noise)
                temp, obj_opt, heterogeneity = main(x0, A, b, comm_rounds=args.comm_rounds, fedAVG_local_epochs=lep, fedAVG_lr=lr)
                heterogeneity_values.append(heterogeneity)
                obj_vals.append(temp - obj_opt.numpy())

        j=0
        for sigma_noise in noise:
            labels.append(r"$\eta = 10^{{{}}}, heterogeneity\approx{:.3f} \times 10^{{{}}}$".format(np.log10(learning_rate[0]).astype(int), heterogeneity_values[j]/1000, "3"))
            j+=1      


    


    #######################################################################
    name = args.method + "_LR_noise"
    name = name.replace('.', '')
    pickle.dump([obj_vals,heterogeneity_values], open(name+'.pkl', 'wb'))
    #######################################################################