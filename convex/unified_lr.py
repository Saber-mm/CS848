import numpy as np
import torch
import copy
import argparse
import torch as torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import math as math
import pickle

def main(A, b, comm_rounds, eta=0.01, proximal_update_epochs=100, proximal_update_lr=0.01, fedAVG_local_epochs=2, fedAVG_lr=0.0001):
    
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
    eta_initial = eta
    model_global = LogisticRegression(A[0].shape[1])
    x_central = model_global.state_dict()['linear.weight'].numpy()
    z_clients = [x_central for c in range(len(A))]
    w_clients = [x_central for c in range(len(A))]
    u_clients = [x_central for c in range(len(A))]

    participation_masks = [np.random.binomial(1, args.participation_prob, len(A)) for e in range(comm_rounds)]
    
    stored_global_U=[]
    stored_global_T=[]

    obj_val = []
    ###########################
    if args.method != 'fedAVG':
        for e in range(comm_rounds):
            ##############################
            if args.eta_diminish == 'log':
                eta = eta_initial/(np.log10(e+1)+1)
            elif args.eta_diminish == 'sqrt':
                eta = eta_initial/np.sqrt(e+1)
            elif args.eta_diminish == 'linear':
                eta = eta_initial/(e+1)
            elif args.eta_diminish == 'exponential':
                T = args.division_period
                if e!=0 and e%T==0:
                    eta = eta/2
            ###############################   
            if e%10==0:
                print("at communication round {}".format(e))

            for j in range(len(A)):
                if participation_masks[e][j] == 1:
                    z_clients[j] = ((1-alpha)*u_clients[j]) + (alpha*prox_update(copy.deepcopy(u_clients[j]), copy.deepcopy(model_global), 
                    A[j], b[j], eta, proximal_update_epochs, proximal_update_lr))
                
            # projecting local solutions on the equality subspace:
            x_central = model_avg(z_clients, participation_masks[e])
            global_model_parameters_dict = model_global.state_dict()
            for k in global_model_parameters_dict.keys():
                if k == 'linear.weight':
                    global_model_parameters_dict[k] = torch.tensor(x_central)
            model_global.load_state_dict(global_model_parameters_dict)        
            
            obj_val.append(objective(copy.deepcopy(model_global), A, b).numpy())
            #######################################
            global_U_round = u_clients[0].squeeze()
            for j in range(1, len(A)):
                global_U_round = np.concatenate((global_U_round, u_clients[j].squeeze()))   
            stored_global_U.append(global_U_round)
            ######################################
            # finding the next iteration proximal point for each user:
            for j in range(len(A)):
                w_clients[j] = ((1-beta)*z_clients[j]) + (beta*x_central)
                u_clients[j] = ((1-gamma)*u_clients[j]) + (gamma*w_clients[j])
            #######################################
            global_T_round = u_clients[0].squeeze()
            for j in range(1, len(A)):
                global_T_round = np.concatenate((global_T_round, u_clients[j].squeeze()))   
            stored_global_T.append(global_T_round)
            ########################################
            if args.global_acceleration == True:
                m, d, Window = 10, 100, args.acc_window
                    
                global_U_matrix = np.zeros((m*d, Window+1))
                c=0
                for global_u in stored_global_U[-(Window+1):]:
                    global_U_matrix[:,c] = global_u
                    c+=1

                global_T_matrix = np.zeros((m*d, Window+1))
                c=0
                for global_t in stored_global_T[-(Window+1):]:
                    global_T_matrix[:,c] = global_t
                    c+=1       
                
                if e>=Window+1:    
                    G=np.matmul((global_U_matrix-global_T_matrix).T, (global_U_matrix-global_T_matrix))
                    numerator = np.matmul(np.linalg.pinv(G), np.ones((Window+1,1)))
                    pi = numerator/np.matmul(np.ones((1,Window+1)), numerator)
                    u_global = np.zeros((m*d))
                    for i in range(Window+1):
                        u_global += pi[i]*global_T_matrix[:,i]
                    for j in range(len(A)):
                        # finding the next iteration proximal point for each user based on the global Anderson acceleration
                        u_clients[j] = u_global[j*d:(j+1)*d]

    #############################
    elif args.method == 'fedAVG':
        for e in range(comm_rounds):
            if e%10==0:
                print("at communication round {}".format(e))
            
            for j in range(len(A)):

                model_j = LogisticRegression(A[0].shape[1])
                model_j_parameters_dict = model_global.state_dict()
                for k in model_j_parameters_dict.keys():
                    if k == 'linear.weight':
                        model_j_parameters_dict[k] = torch.reshape(torch.tensor(u_clients[j]), (1,A[0].shape[1]))
                model_j.load_state_dict(model_j_parameters_dict)  
                
                if participation_masks[e][j] == 1:
                    z_clients[j] = ((1-alpha)*u_clients[j]) + (alpha*fedAVG_model_update(copy.deepcopy(model_j), A[j], b[j], fedAVG_local_epochs, fedAVG_lr))
            
            # projecting local solutions on the equality subspace:
            x_central = model_avg(z_clients, participation_masks[e])
            global_model_parameters_dict = model_global.state_dict()
            for k in global_model_parameters_dict.keys():
                if k == 'linear.weight':
                    global_model_parameters_dict[k] = torch.tensor(x_central)
            model_global.load_state_dict(global_model_parameters_dict)        
            
            obj_val.append(objective(copy.deepcopy(model_global), A, b).numpy())
            #######################################
            global_U_round = u_clients[0].squeeze()
            for j in range(1, len(A)):
                global_U_round = np.concatenate((global_U_round, u_clients[j].squeeze()))   
            stored_global_U.append(global_U_round)
            ######################################
            # finding the next iteration proximal point for each user:
            for j in range(len(A)):
                w_clients[j] = ((1-beta)*z_clients[j]) + (beta*x_central)
                u_clients[j] = ((1-gamma)*u_clients[j]) + (gamma*w_clients[j])
            #######################################
            global_T_round = u_clients[0].squeeze()
            for j in range(1, len(A)):
                global_T_round = np.concatenate((global_T_round, u_clients[j].squeeze()))   
            stored_global_T.append(global_T_round)
            ########################################

            if args.global_acceleration == True:
                m, d, Window = 10, 100, args.acc_window
                    
                global_U_matrix = np.zeros((m*d, Window+1))
                c=0
                for global_u in stored_global_U[-(Window+1):]:
                    global_U_matrix[:,c] = global_u
                    c+=1

                global_T_matrix = np.zeros((m*d, Window+1))
                c=0
                for global_t in stored_global_T[-(Window+1):]:
                    global_T_matrix[:,c] = global_t
                    c+=1       
                
                if e>=Window+1:    
                    G=np.matmul((global_U_matrix-global_T_matrix).T, (global_U_matrix-global_T_matrix))
                    numerator = np.matmul(np.linalg.pinv(G), np.ones((Window+1,1)))
                    pi = numerator/np.matmul(np.ones((1,Window+1)), numerator)
                    u_global = np.zeros((m*d))
                    for i in range(Window+1):
                        u_global += pi[i]*global_T_matrix[:,i]
                    for j in range(len(A)):
                        # finding the next iteration proximal point for each user based on the global Anderson acceleration
                        u_clients[j] = u_global[j*d:(j+1)*d]

    return np.array(obj_val)




def objective(model, A, b):
    m = len(A)
    model.eval()
    with torch.no_grad():
        obj = 0
        for i in range(m):
            obj += criterion(b[i], A[i], model)
    return obj / m


def prox_update(prox_point_j, model, aj, bj, eta, proximal_update_epochs, proximal_update_lr):
    
    optimizer = torch.optim.SGD(model.parameters(), lr=proximal_update_lr)
    proximal_point_j = torch.from_numpy(prox_point_j)

    if args.use_SGD:

        batch_size = args.batch_size
        model.train()
        for t in range(proximal_update_epochs):
            index = np.arange(aj.shape[0])
            np.random.shuffle(index)
            aj_shuffled = aj[index,:]
            bj_shuffled = bj[index]
            for i in range(np.int(aj.shape[0]/batch_size)):
                model.zero_grad()
                aj_batch = aj_shuffled[i*batch_size:(i+1)*batch_size,:]
                bj_batch = bj_shuffled[i*batch_size:(i+1)*batch_size]

                fj = criterion(bj_batch, aj_batch, model)
                if eta == np.inf:
                    proximalmap_objective = fj.double()
                elif eta != np.inf:
                    model_parameters = torch.transpose(model.linear.weight, 0, 1)
                    proximal_regularizer = (torch.norm(proximal_point_j.double()-torch.transpose(model_parameters.double(),0,1))**2)/(2*eta)
                    proximalmap_objective = fj.double() + proximal_regularizer

                proximalmap_objective.backward()
                optimizer.step()

        return  model.state_dict()['linear.weight'].numpy()


    else:
        
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

    if args.use_SGD:

        batch_size = args.batch_size
        model.train()
        for t in range(l_ep):
            index = np.arange(A.shape[0])
            np.random.shuffle(index)
            A_shuffled = A[index,:]
            b_shuffled = b[index]
            for i in range(np.int(A.shape[0]/batch_size)):
                model.zero_grad()
                A_batch = A_shuffled[i*batch_size:(i+1)*batch_size,:]
                b_batch = b_shuffled[i*batch_size:(i+1)*batch_size]
                loss = criterion(b_batch, A_batch, model)
                loss.backward()
                optimizer.step()

        return model.state_dict()['linear.weight'].numpy()

    else:

        model.train()
        for t in range(l_ep):
            model.zero_grad()
            loss = criterion(b, A, model)
            loss.backward()
            optimizer.step()

        return model.state_dict()['linear.weight'].numpy()


    


def model_avg(models, participation_mask):
    x = np.zeros(models[0].shape)
    for i in range(len(models)):
        if participation_mask[i] == 1:
            x += models[i]
    return x/sum(participation_mask)


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

def data_gen(m=10, d=100, n=1000, sigma_a=1):
    x0 = torch.randn(d).reshape(-1, 1)
    A = []
    b = []
    for j in range(m):
        a = torch.randn(n*d).reshape(n, d)
        A.append(a)
        dis = torch.matmul(A[j], x0)
        prob = torch.sigmoid(dis)# - 0.5
        b.append((2*torch.bernoulli(prob))-1)

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
    parser.add_argument('--participation_prob', type=float, default=1.0, help='The prob that a user participates in a round')
    parser.add_argument('--use_SGD', type=bool, default=False, help="whether clients use SGD for their local computation or not")
    parser.add_argument('--batch_size', type=int, default=200, help='size of the batches for local SGD')
    parser.add_argument('--global_acceleration', type=bool, default=False, help="whether use Anderson global acceleration or not")
    parser.add_argument('--acc_window', type=int, default=2, help="acceleration window size")
    ###
    parser.add_argument('--prox_lep', type=int, default=100, help='epcohs for proximal operator')
    parser.add_argument('--prox_lr', type=float, default=0.01, help='learning rate for pSroximal operator (for FedProx, in case it returns Nan, use smaller values like 0.001)')
    ###
    parser.add_argument('--eta_stepsize', type=float, default=0.01, help='proximal mapping step size (recommended value for FedProx: 0.001, for rest: 0.01)')
    parser.add_argument('--eta_diminish', type=str, default='None', help="method for diminishing eta, in {None, log, sqrt, exponential, linear}")
    parser.add_argument('--division_period', type=int, default=500, help='period for dividing eta by 2, when eta_diminish==exponential')
    parser.add_argument('--fedAVG_lep', type=int, default=5, help='local epochs for fedAVG')
    parser.add_argument('--fedAVG_lr', type=float, default=0.01, help='local learning rate for fedAVG')
    parser.add_argument('--option', type=int, default=0, help="option in {1, 2, 3, 22} for fedAVG")
    parser.add_argument('--seed', type=int, default=250, help='random seed')
    m, n, d = 10, 1000, 100  # number of users, number of points of each user, dimensionality 

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x0, A, b = data_gen(m, d, n)
    model = LogisticRegression(d)
    print('finding the global optimizer using GD ...')
    x_opt, obj_opt = minimiser(A, b, lr=0.001, epochs=5000)
    print((0.001,obj_opt))
    
    ###########################
    if args.method != 'fedAVG':
        Eta = args.eta_stepsize * np.array([1])
        labels = []
        idx = [1]
        for k in idx:
            if k==1:
                labels.append(r"$\eta={}$".format(args.eta_stepsize))
            elif k==np.inf:
                labels.append(r"$\eta=\infty$")
            else:
                labels.append(r"$\eta={}*{}$".format(args.eta_stepsize, k))

        obj_vals = []
        for eta in Eta:
            print("eta: {}".format(eta))
            temp = main(A, b, comm_rounds=args.comm_rounds, eta=eta, proximal_update_epochs=args.prox_lep, proximal_update_lr=args.prox_lr)
            obj_vals.append(temp - obj_opt.numpy())  



    #############################``
    elif args.method == 'fedAVG':
        if args.option == 1:
            learning_rate = args.fedAVG_lr * np.array([1, 1/2, 1/5, 1/10, 1/20, 1/50])
            local_epoch = [localepoch] * len(learning_rate)
            labels = []
            idx = [1, 2, 5, 10, 20, 50]
            for i, j, l in zip(local_epoch, learning_rate, idx):
                if l > 1:
                    labels.append(r"k={}, $\eta={}/{}$".format(i, args.fedAVG_lr, l))
                else:
                    labels.append(r"k={}, $\eta={}$".format(i, args.fedAVG_lr))
        elif args.option == 2:
            local_epoch = [2, 5, 10, 20, 50]
            learning_rate = args.fedAVG_lr / (np.array(local_epoch)-1)
            labels = []
            for i, j in zip(local_epoch, learning_rate):
                labels.append(r"k={}, $\eta={}/(k-1)$".format(i, args.fedAVG_lr))
        elif args.option == 3:
            #local_epoch = [1, 2, 5, 10, 20, 50]
            local_epoch = [args.fedAVG_lep]
            learning_rate = [args.fedAVG_lr] * len(local_epoch)
            labels = []
            for i, j in zip(local_epoch, learning_rate):
                labels.append(r"k={}, $\eta$={}".format(i, args.fedAVG_lr))
        else:
            local_epoch = [args.fedAVG_lep]
            learning_rate = [args.fedAVG_lr]

        obj_vals = []
        for lep, lr in zip(local_epoch, learning_rate):
            print(lep, lr)
            temp = main(A, b, comm_rounds=args.comm_rounds, fedAVG_local_epochs=lep, fedAVG_lr=lr)
            obj_vals.append(temp - obj_opt.numpy())  


    ######################################################################
    name = args.method + "_LR"
    name = name.replace('.', '')
    pickle.dump(obj_vals, open(name+'.pkl', 'wb'))
    #######################################################################