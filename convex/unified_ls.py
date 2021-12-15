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
    x_central = np.zeros(x0.shape)
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
                if eta != np.inf:
                    if participation_masks[e][j] == 1:
                        z_clients[j] = ((1-alpha)*u_clients[j]) + (alpha*prox_update(copy.deepcopy(u_clients[j]), A[j], b[j], eta))
                elif eta == np.inf:
                    if participation_masks[e][j] == 1:
                        z_clients[j] = ((1-alpha)*u_clients[j]) + (alpha*prox_update_etainf(copy.deepcopy(u_clients[j]), A[j], b[j], eta))
            # projecting local solutions on the equality subspace:
            x_central = model_avg(z_clients, participation_masks[e])
            obj_val.append(objective(x_central, A, b))
            #######################################
            global_U_round = u_clients[0]
            for j in range(1, len(A)):
                global_U_round = np.concatenate((global_U_round, u_clients[j]))
            stored_global_U.append(global_U_round)
            ######################################
            # finding the next iteration proximal point for each user:
            for j in range(len(A)):
                w_clients[j] = ((1-beta)*z_clients[j]) + (beta*x_central)
                u_clients[j] = ((1-gamma)*u_clients[j]) + (gamma*w_clients[j])
            #######################################
            global_T_round = u_clients[0]
            for j in range(1, len(A)):
                global_T_round = np.concatenate((global_T_round, u_clients[j]))   
            stored_global_T.append(global_T_round)
            ########################################
            # 1)        
            if args.global_acceleration == True:
                m, d, Window = 25, 100, args.acc_window
                    
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
                if participation_masks[e][j] == 1:
                    z_clients[j] = ((1-alpha)*u_clients[j]) + (alpha*fedAVG_model_update(copy.deepcopy(u_clients[j]), A[j], b[j], fedAVG_local_epochs, fedAVG_lr))
            
            # projecting local solutions on the equality subspace:
            x_central = model_avg(z_clients, participation_masks[e])
            obj_val.append(objective(x_central, A, b))
            #######################################
            global_U_round = u_clients[0]
            for j in range(1, len(A)):
                global_U_round = np.concatenate((global_U_round, u_clients[j]))   
            stored_global_U.append(global_U_round)
            ######################################
            # finding the next iteration proximal point for each user:
            for j in range(len(A)):
                w_clients[j] = ((1-beta)*z_clients[j]) + (beta*x_central)
                u_clients[j] = ((1-gamma)*u_clients[j]) + (gamma*w_clients[j])
            #######################################
            global_T_round = u_clients[0]
            for j in range(1, len(A)):
                global_T_round = np.concatenate((global_T_round, u_clients[j]))   
            stored_global_T.append(global_T_round)
            ########################################
     
            if args.global_acceleration == True:
                m, d, Window = 25, 100, args.acc_window
                    
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

    return obj_val, obj_opt    


def objective(x, A, b):
    f = 0
    for i in range(len(A)):
        f += 0.5 * np.linalg.norm(np.matmul(A[i], x) - b[i])**2
    return f


def prox_update(proximal_point_j, Aj, bj, eta):

    if args.use_SGD:
        batch_size = args.batch_size
        lr = args.prox_lr
        x = proximal_point_j
        
        for i in range(args.prox_lep):
            index = np.arange(Aj.shape[0])
            np.random.shuffle(index)
            Aj_shuffled = Aj[index,:]
            bj_shuffled = bj[index]
            for j in range(np.int(Aj.shape[0]/batch_size)):
                A_batch = Aj_shuffled[j*batch_size:(j+1)*batch_size,:]
                b_batch = bj_shuffled[j*batch_size:(j+1)*batch_size]

                m = np.matmul(A_batch.T, A_batch) + (np.eye(A_batch.shape[1])/eta)
                n = np.matmul(A_batch.T, b_batch) + (proximal_point_j/eta)
                g = np.matmul(m, x) - n
                x = x - lr*g

        return x

    else:
        m = np.matmul(Aj.T, Aj) + (np.eye(Aj.shape[1])/eta)
        n = np.matmul(Aj.T, bj) + (proximal_point_j/eta)

        return np.matmul(np.linalg.inv(m), n)    

def prox_update_etainf(proximal_point_j, Aj, bj):

    if args.use_SGD:
        batch_size = args.batch_size
        lr = args.prox_lr
        x = proximal_point_j
        
        for i in range(args.prox_lep):
            index = np.arange(Aj.shape[0])
            np.random.shuffle(index)
            Aj_shuffled = Aj[index,:]
            bj_shuffled = bj[index]
            for j in range(np.int(Aj.shape[0]/batch_size)):
                A_batch = Aj_shuffled[j*batch_size:(j+1)*batch_size,:]
                b_batch = bj_shuffled[j*batch_size:(j+1)*batch_size]

                m = np.matmul(A_batch.T, A_batch)
                n = np.matmul(A_batch.T, b_batch)
                g = np.matmul(m, x) - n
                x = x - lr*g

        return x

    else:
        m = np.matmul(Aj.T, Aj)
        n = np.matmul(Aj.T, bj)

        return np.matmul(np.linalg.inv(m), n)    


def fedAVG_model_update(x, A, b, l_ep, lr):

    if args.use_SGD:

        batch_size = args.batch_size
        lr = args.fedAVG_lr
        
        for i in range(l_ep):
            index = np.arange(A.shape[0])
            np.random.shuffle(index)
            A_shuffled = A[index,:]
            b_shuffled = b[index]
            for j in range(np.int(A.shape[0]/batch_size)):
                A_batch = A_shuffled[j*batch_size:(j+1)*batch_size,:]
                b_batch = b_shuffled[j*batch_size:(j+1)*batch_size]

                r = np.matmul(A_batch, x) - b_batch
                g = np.matmul(A_batch.T, r)
                x = x - lr * g

        return x

    else:
        
        for i in range(l_ep):
            r = np.matmul(A, x) - b
            g = np.matmul(A.T, r)
            x = x - lr * g
        return x   


def model_avg(models, participation_mask):

    x = np.zeros(models[0].shape)
    for i in range(len(models)):
        if participation_mask[i] == 1:
            x += models[i]

    return x/sum(participation_mask)



def minimiser(A, b):

    n, d = A[0].shape
    A_sq = np.zeros((d, d))
    Ab = np.zeros(d)
    for i in range(len(A)):
        A_sq += np.matmul(A[i].T, A[i])
        Ab += np.matmul(A[i].T, b[i])
    A_sq_inv = np.linalg.inv(A_sq)

    return np.matmul(A_sq_inv, Ab)


def model_gen(m=25, d=100, n=5000):

    sigma_a = 1
    sigma_noise = 0.25
    np.random.seed(100)
    x0 = np.random.randn(d)
    A = []
    b = []
    for i in range(m):
        a = np.random.normal(0, sigma_a, n*d)    
        A.append(a.reshape(n, d))
        Vi = np.random.normal(0, sigma_noise, n)
        b.append(np.matmul(A[i], x0)+Vi)

    return x0, A, b


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--comm_rounds', type=int, default=605, help="number of rounds of training")
    parser.add_argument('--method', type=str, default='fedRP', help="method in {fedAVG, fedProx, fedSplit, fedPI, fedRP}")
    parser.add_argument('--participation_prob', type=float, default=1.0, help='The prob that a user participates in a round')
    parser.add_argument('--use_SGD', type=bool, default=False, help="whether clients use SGD for their local computation or not")
    parser.add_argument('--batch_size', type=int, default=1000, help='size of the batches for local SGD')
    parser.add_argument('--global_acceleration', type=bool, default=False, help="whether use Anderson global acceleration or not")
    parser.add_argument('--acc_window', type=int, default=2, help="acceleration window size")
    ###
    parser.add_argument('--prox_lep', type=int, default=100, help='epcohs for proximal operator')
    parser.add_argument('--prox_lr', type=float, default=0.0000001, help='learning rate for pSroximal operator (recommended value around: 0.001 (the smallet the eta_stepsize the smaller prox_lr))')
    ###
    parser.add_argument('--eta_stepsize', type=float, default=0.00001, help='proximal mapping step size')
    parser.add_argument('--eta_diminish', type=str, default='None', help="method for diminishing eta, in {None, log, sqrt, exponential, linear}")
    parser.add_argument('--division_period', type=int, default=500, help='period for dividing eta by 2, when eta_diminish==exponential')
    parser.add_argument('--fedAVG_lep', type=int, default=5, help='local epochs for fedAVG')
    parser.add_argument('--fedAVG_lr', type=float, default=0.00001, help='local learning rate for fedAVG')
    parser.add_argument('--option', type=int, default=0, help="option in {1, 2, 3, 22} for fedAVG")
    parser.add_argument('--seed', type=int, default=345, help='random seed')


    args = parser.parse_args()
    np.random.seed(args.seed)
    x0, A, b = model_gen()
    
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
            temp, obj_opt = main(x0, A, b, comm_rounds=args.comm_rounds, eta=eta)
            obj_vals.append(temp - obj_opt)


    #############################
    elif args.method == 'fedAVG':
        if args.option == 1:
            learning_rate = args.fedAVG_lr * np.array([1, 1/2, 1/5, 1/10, 1/20, 1/50])
            local_epoch = [args.fedAVG_lep] * len(learning_rate)
            labels = []
            idx = [1, 2, 5, 10, 20, 50]
            for i, j, l in zip(local_epoch, learning_rate, idx):
                if l > 1:
                    labels.append(r"k={}, $\eta={}/{}$".format(i, args.fedAVG_lr, l))
                else:
                    labels.append(r"k={}, $\eta={}$".format(i, args.fedAVG_lr))
        elif args.option == 2:
            local_epoch = [2, 5, 10, 20, 50, 100]
            learning_rate = args.fedAVG_lr / (np.array(local_epoch)-1)
            labels = []
            for i, j in zip(local_epoch, learning_rate):
                labels.append(r"k={}, $\eta={}/(k-1)$".format(i, args.fedAVG_lr))
        elif args.option == 22:
            local_epoch = [2, 3, 4, 5, 10, 20, 50, 100]
            learning_rate = args.fedAVG_lr / (np.array(local_epoch)-1)**2
            labels = []
            for i, j in zip(local_epoch, learning_rate):
                labels.append(r"k={}, $\eta={}/(k-1)^2$".format(i, args.fedAVG_lr))
        elif args.option == 3:
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
            temp, obj_opt = main(x0, A, b, comm_rounds=args.comm_rounds, fedAVG_local_epochs=lep, fedAVG_lr=lr)
            obj_vals.append(temp - obj_opt)




    name = args.method + "_LS"
    name = name.replace('.', '')
    pickle.dump(obj_vals, open(name+'.pkl', 'wb'))
    #######################################################