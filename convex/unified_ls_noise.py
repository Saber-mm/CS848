import numpy as np
import copy
import argparse
import math
import pickle as pickle
from numpy import linalg as LA

def main(x0, A, b, comm_rounds=500,  eta=0.001, fedAVG_local_epochs=400, fedAVG_lr=0.0001):

    np.random.seed(100)

    x_orig = copy.deepcopy(x0)
    x_opt = minimiser(A, b)
    obj_opt = objective(x_opt, A, b)
    heterogeneity = heterogeneity_measure(x_opt, A, b)
    
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
    x_central = np.zeros(x0.shape)
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
                if eta != np.inf:
                    z_clients[j] = ((1-alpha)*u_clients[j]) + (alpha*prox_update(copy.deepcopy(u_clients[j]), A[j], b[j], eta))
                elif eta == np.inf:
                    z_clients[j] = ((1-alpha)*u_clients[j]) + (alpha*prox_update_etainf(copy.deepcopy(u_clients[j]), A[j], b[j], eta))
            # projecting local solutions on the equality subspace:
            x_central = model_avg(z_clients)
            obj_val.append(objective(x_central, A, b))
            
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
                z_clients[j] = ((1-alpha)*u_clients[j]) + (alpha*fedAVG_model_update(copy.deepcopy(u_clients[j]), A[j], b[j], fedAVG_local_epochs, fedAVG_lr))
            
            # projecting local solutions on the equality subspace:
            x_central = model_avg(z_clients)
            obj_val.append(objective(x_central, A, b))
            
            # finding the next iteration proximal point for each user:
            for j in range(len(A)):
                w_clients[j] = ((1-beta)*z_clients[j]) + (beta*x_central)
                u_clients[j] = ((1-gamma)*u_clients[j]) + (gamma*w_clients[j])        
    ###############################


    return obj_val, obj_opt, heterogeneity    


def objective(x, A, b):
    f = 0
    for i in range(len(A)):
        f += 0.5 * np.linalg.norm(np.matmul(A[i], x) - b[i])**2
    return f


def heterogeneity_measure(x_opt, A, b):
    sum_gradient_norm_2 = 0
    
    for i in range(len(A)):
        A_sq = np.matmul(A[i].T, A[i])
        Ab = np.matmul(A[i].T, b[i])
        client_i_gradient = 2*np.matmul(A_sq, x_opt)-(2*Ab)
        sum_gradient_norm_2 += np.power(np.linalg.norm(client_i_gradient), 2)
    
    return sum_gradient_norm_2/len(A)   



def prox_update(proximal_point_j, Aj, bj, eta):
    
    m = np.matmul(Aj.T, Aj) + (np.eye(Aj.shape[1])/eta)
    n = np.matmul(Aj.T, bj) + (proximal_point_j/eta)

    return np.matmul(np.linalg.inv(m), n)

def prox_update_etainf(proximal_point_j, Aj, bj):
    
    m = np.matmul(Aj.T, Aj)
    n = np.matmul(Aj.T, bj)

    return np.matmul(np.linalg.inv(m), n)

def fedAVG_model_update(x, A, b, l_ep, lr):

    obj0 = 0.5 * np.linalg.norm(np.matmul(A, x)-b)**2
    for i in range(l_ep):
        r = np.matmul(A, x) - b
        g = np.matmul(A.T, r)
        x = x - lr * g
    obj1 = 0.5 * np.linalg.norm(np.matmul(A, x)-b)**2

    return x    


def model_avg(models):

    x = np.zeros(models[0].shape)
    for i in range(len(models)):
        x += models[i]

    return x/len(models)


def minimiser(A, b):

    n, d = A[0].shape
    A_sq = np.zeros((d, d))
    Ab = np.zeros(d)
    for i in range(len(A)):
        A_sq += np.matmul(A[i].T, A[i])
        Ab += np.matmul(A[i].T, b[i])
    A_sq_inv = np.linalg.inv(A_sq)

    return np.matmul(A_sq_inv, Ab)


def model_gen(sigma_noise, m=25, d=100, n=5000):

    np.random.seed(args.seed)
    sigma_a = 1
    np.random.seed(100)
    x0 = np.random.randn(d)
    A = []
    b = []
    for i in range(m):
        a = np.random.normal(0, sigma_a, n*d)
        A.append(a.reshape(n, d))
    for i in range(m):
        Vi = np.random.normal(0, sigma_noise, n)
        b.append(np.matmul(A[i], x0)+Vi)    
    return x0, A, b


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--comm_rounds', type=int, default=605, help="number of rounds of training")
    parser.add_argument('--method', type=str, default='fedAVG', help="method in {fedAVG, fedProx, fedSplit, fedPI, fedRP}")
    parser.add_argument('--eta_stepsize', type=float, default=0.00001, help='proximal mapping step size')
    parser.add_argument('--fedAVG_lep', type=int, default=1, help='local epochs for fedAVG')
    parser.add_argument('--fedAVG_lr', type=float, default=0.00001, help='local learning rate for fedAVG')
    parser.add_argument('--option', type=int, default=3, help="option in {1, 2, 3, 22} for fedAVG")
    parser.add_argument('--seed', type=int, default=345, help='random seed')


    args = parser.parse_args()
    np.random.seed(args.seed)
    show = False
    comm_rounds = args.comm_rounds
    noise = [0.25, 2, 10]
    
    ###########################
    if args.method != 'fedAVG':
        Eta = args.eta_stepsize * np.array([1])
        idx = [1]

        labels = []
        obj_vals = []
        heterogeneity_values = []
        
        for eta in Eta:
            for sigma_noise in noise:
                print("sigma noise: {}".format(sigma_noise))
                x0, A, b = model_gen(sigma_noise)
                temp, obj_opt, heterogeneity = main(x0, A, b, comm_rounds=args.comm_rounds, eta=eta)
                heterogeneity_values.append(heterogeneity)
                obj_vals.append(temp - obj_opt)

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
            for sigma_noise in noise:
                print("sigma noise: {}".format(sigma_noise))
                x0, A, b = model_gen(sigma_noise)
                temp, obj_opt, heterogeneity = main(x0, A, b, comm_rounds=args.comm_rounds, fedAVG_local_epochs=lep, fedAVG_lr=lr)
                heterogeneity_values.append(heterogeneity)
                obj_vals.append(temp - obj_opt)

        j=0
        for sigma_noise in noise:
            labels.append(r"$\eta = 10^{{{}}}, heterogeneity\approx{:.3f} \times 10^{{{}}}$".format(np.log10(learning_rate[0]).astype(int), heterogeneity_values[j]/1000, "3"))
            j+=1        



    #######################################################
    name = args.method + "_LS_noise"
    name = name.replace('.', '')
    pickle.dump([obj_vals,heterogeneity_values], open(args.method+'_LS_noisek=1.pkl', 'wb'))
    #######################################################