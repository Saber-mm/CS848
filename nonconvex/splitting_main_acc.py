import os
import copy
import time
import numpy as np
import torch
from tensorboardX import SummaryWriter
from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, Vgg9, CNNEmnist, CNNCifarSmall
from utils import get_dataset, exp_details, user_inference
from splitting_server_algo import fed_average, peaceman_rachford, douglas_rachford, reflection_projection, anderson_acceleration


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(2)
    start_time = time.time()
    print_every = 5
    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    args = args_parser()
    exp_details(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    filename = \
        "save/{}/{}_N{}_EPS{}_{}_{}_nuser{}_T{}_C{}_iid{}_MC_{}_E{}_B{}_Bfull{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}" \
        "_accel{}_win{}_acceavg{}_reuse{}_seed{}".format(args.save_dir, args.method, args.normalize, args.epsilon, args.dataset,
                                                args.model, args.num_users, args.epochs, args.frac, args.iid, args.user_max_class, args.local_ep,
                                                args.local_bs, args.full_batch, args.lr, args.lr_serv, args.decay_rate,
                                                args.prox_weight, args.qffl, args.Lipschitz_constant,
                                                args.acceleration, args.memory, args.acc_avg, args.reuse, args.seed)
    path = "add the path to a saved model"

    npz_filename = filename + ".npz"

    path = path + ".pt"
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    args.num_users = len(user_groups)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
        elif args.dataset == "emnist":
            global_model = CNNEmnist(args=args)
    elif args.model == "cnnsmall":
        if args.dataset == 'cifar':
            global_model = CNNCifarSmall(args=args)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    elif args.model == 'vgg9':
        global_model = Vgg9(args=args)
        print("hello to VGG-9")
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    if args.reuse == "True":
        global_model.load_state_dict(torch.load(path))

    local_input_model = [copy.deepcopy(global_model) for _ in range(args.num_users)]
    weight_memory_u,  weight_memory_T= [], []
    local_z_model = [copy.deepcopy(local_input_model[i].state_dict()) for i in range(args.num_users)]  # internal models for splitting algorithms
    local_z_model_old = copy.deepcopy(local_z_model)  # internal models old for DR
    model_total_params = [p.numel() for p in global_model.parameters() if p.requires_grad]
    print('model trainable parameter count: sum{}={}\n'.format(model_total_params, sum(model_total_params)))

    total_train_loss, total_train_accuracy = [], []
    total_test_loss, total_test_accuracy = [], []
    users_test_accuracy, users_test_loss = [], []
    users_train_accuracy, users_train_loss = [], []

    users_test_accuracy_his, users_test_loss_his = [], []
    users_train_accuracy_his, users_train_loss_his = [], []

    global_model_history = []

    # Training
    start_time_training = time.time()
    total_time = [start_time_training]
    for epoch in range(args.epochs):
        print(f'\n | Global Training Round : {epoch + 1} |\n')
        local_weights, local_losses, local_norms = [], [], []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users.sort()

        # Update the local models
        # weight_memory_u.append(copy.deepcopy(local_input_model))
        weight_memory_u.append(copy.deepcopy(global_model))
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            w, loss, loss_post, accuracy_post, norm = local_model.update_weights(
                model=copy.deepcopy(local_input_model[idx]), global_round=epoch, user_index=idx)
            local_weights.append(copy.deepcopy(w))

        if args.method == "fedavg":
            global_model, local_input_model = fed_average(copy.deepcopy(local_weights), copy.deepcopy(global_model))
        elif args.method == "fedDR":
            global_model, local_input_model, local_z_model = douglas_rachford(
                copy.deepcopy(local_weights), copy.deepcopy(global_model), copy.deepcopy(local_z_model), copy.deepcopy(local_z_model_old))
            local_z_model_old = copy.deepcopy(local_weights)
        elif args.method == "fedPR":
            print("FedSplit")
            global_model, local_input_model, local_z_model = peaceman_rachford(
                copy.deepcopy(local_weights), copy.deepcopy(global_model), copy.deepcopy(local_z_model))
        elif args.method == "fedRP":
            global_model, local_input_model = reflection_projection(copy.deepcopy(local_weights), copy.deepcopy(global_model))
        total_time.append(time.time()-start_time_training)
        weight_memory_T.append(copy.deepcopy(global_model))
        # weight_memory.append(copy.deepcopy(local_input_model))

        # Acceleration
        if len(weight_memory_T) > args.memory:
            weight_memory_T = weight_memory_T[-args.memory:]
            weight_memory_u = weight_memory_u[-args.memory:]
        if args.acceleration == 1 and epoch >= args.memory:
            global_model = anderson_acceleration_v3(copy.deepcopy(weight_memory_u), copy.deepcopy(weight_memory_T), device, args)
            if args.method == "fedDR":
                global_model, local_input_model, local_z_model = douglas_rachford(
                    copy.deepcopy(local_weights), copy.deepcopy(global_model), copy.deepcopy(local_z_model), copy.deepcopy(local_z_model_old))
            elif args.method == "fedPR":
                global_model, local_input_model, local_z_model = peaceman_rachford(
                    copy.deepcopy(local_weights), copy.deepcopy(global_model), copy.deepcopy(local_z_model))
            elif args.method == "fedRP":
                global_model, local_input_model = reflection_projection(copy.deepcopy(local_weights), copy.deepcopy(global_model))
            else:
                local_input_model = [copy.deepcopy(global_model) for _ in range(args.num_users)]

        # Do an inference on selected participants to get UPDATED training loss, training accuracy, testing accuracy
        user_train_accuracy, user_train_loss, user_test_accuracy, user_test_loss, user_size = user_inference(
            idxs_users, global_model, LocalUpdate, args, train_dataset, user_groups, logger)
        users_test_accuracy.append(user_test_accuracy)
        users_test_loss.append(user_test_loss)
        users_train_accuracy.append(user_train_accuracy)
        users_train_loss.append(user_train_loss)

        # History
        # global_model_history.append(copy.deepcopy(global_model.state_dict()))
        # win_len = 5
        # if len(global_model_history) > win_len:
        #     global_model_history = global_model_history[-win_len:]
        # print(len(global_model_history))
        # global_avg_his, _ = fed_average(copy.deepcopy(global_model_history), copy.deepcopy(global_model))
        # user_train_accuracy, user_train_loss, user_test_accuracy, user_test_loss, user_size = user_inference(
        #     idxs_users, global_avg_his, LocalUpdate, args, train_dataset, user_groups, logger)
        # users_test_accuracy_his.append(user_test_accuracy)
        # users_test_loss_his.append(user_test_loss)
        # users_train_accuracy_his.append(user_train_accuracy)
        # users_train_loss_his.append(user_train_loss)

        # print(sum(users_train_loss_his[-1]), sum(users_train_loss[-1]))

        print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(users_train_loss))}')
        print('Train Accuracy: {:.2f}% \n'.format(100*np.mean(np.array(users_train_accuracy))))

        if epoch % print_every == 0:
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            total_test_accuracy.append(test_acc)
            total_test_loss.append(test_acc)
            print(test_acc)
        if (epoch+1) % 50 == 0:
            pt_filename = \
                "save/{}/method_{}_N{}_EPS{}_{}_{}_nuser{}_epochs{}_C{}_iid{}_maxclass_{}_E{}_B{}_Bfull{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}" \
                "_vip{}_vscale{}_vbias{}_accel{}_win{}_acceavg{}_reuse{}_seed{}".format(args.save_dir, args.method, args.normalize, args.epsilon, args.dataset,
                                                                              args.model, args.num_users, epoch, args.frac, args.iid, args.user_max_class, args.local_ep,
                                                                              args.local_bs, args.full_batch, args.lr, args.lr_serv, args.decay_rate,
                                                                              args.prox_weight, args.qffl, args.Lipschitz_constant,
                                                                              args.vip, args.vip_scale, args.vip_bias, args.acceleration, args.memory,
                                                                                        args.acc_avg, args.reuse, args.seed)
            pt_filename = pt_filename + ".pt"
            torch.save(global_model.state_dict(), pt_filename)
    print("time{}:".format(time.time()-start_time))

    np.savez(npz_filename, acc=[total_test_accuracy], train_acc=[total_test_loss],
             user=[users_test_accuracy, users_test_loss, users_train_accuracy, users_train_loss],
             user_his=[users_test_accuracy_his, users_test_loss_his, users_train_accuracy_his, users_train_loss_his],
             time=[total_time])







