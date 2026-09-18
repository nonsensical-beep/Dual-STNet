import argparse
import cv2
from os.path import isdir, join
import os

import numpy as np

from model.vitCA_and_MutiScale_ModernTCN2D import *
from torch.utils import data
from core.functions import *
from torch import nn, optim
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

random_seed = 30
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings('ignore')

k_size = 399


def get_data(args, test=False):
    seismic_data = np.load('./data/marmousi2_seismic_3.npy')  # (3400,3,1201)
    seismic_data = np.delete(seismic_data, 0, axis=2)  # (3400,3,1200)

    data_vs = np.load('./data/marmousi2_vs.npy')  # (1201,3400)
    data_vp = np.load('./data/marmousi2_vp.npy')  # (1201,3400)
    data_den = np.load('./data/marmousi2_rho.npy')  # (1201,3400)

    # vs
    init_vs = cv2.GaussianBlur(data_vs, (k_size, k_size), 0)  # (1201,3400)
    init_vs = init_vs[:, np.newaxis]  # (1201,1,3400)
    init_vs = np.swapaxes(init_vs, 0, 2)  # (3400,1,1201)
    init_vs = np.delete(init_vs, 0, axis=2)  # (3400,1,1200) ndarry
    init_vs_mean = torch.tensor(np.mean(init_vs, keepdims=True)).float()
    init_vs_std = torch.tensor(np.std(init_vs, keepdims=True)).float()
    init_vs = torch.tensor(init_vs).float()
    init_vs = init_vs.cuda()
    init_vs_mean = init_vs_mean.cuda()
    init_vs_std = init_vs_std.cuda()
    init_vs_normalization = Normalization(mean_val=init_vs_mean, std_val=init_vs_std)
    init_vs = init_vs_normalization.normalize(init_vs)  # (3400,1,1200)tensor
    # vp
    init_vp = cv2.GaussianBlur(data_vp, (k_size, k_size), 0)  # (1201,3400)
    init_vp = init_vp[:, np.newaxis]  # (1201,1,3400)
    init_vp = np.swapaxes(init_vp, 0, 2)  # (3400,1,1201)
    init_vp = np.delete(init_vp, 0, axis=2)  # (3400,1,1200) ndarry
    init_vp_mean = torch.tensor(np.mean(init_vp, keepdims=True)).float()
    init_vp_std = torch.tensor(np.std(init_vp, keepdims=True)).float()
    init_vp = torch.tensor(init_vp).float()
    init_vp = init_vp.cuda()
    init_vp_mean = init_vp_mean.cuda()
    init_vp_std = init_vp_std.cuda()
    init_vp_normalization = Normalization(mean_val=init_vp_mean, std_val=init_vp_std)
    init_vp = init_vp_normalization.normalize(init_vp)  # (3400,1,1200)tensor
    # den
    init_den = cv2.GaussianBlur(data_den, (k_size, k_size), 0)  # (1201,3400)
    init_den = init_den[:, np.newaxis]  # (1201,1,3400)
    init_den = np.swapaxes(init_den, 0, 2)  # (3400,1,1201)
    init_den = np.delete(init_den, 0, axis=2)  # (3400,1,1200) ndarry
    init_den_mean = torch.tensor(np.mean(init_den, keepdims=True)).float()
    init_den_std = torch.tensor(np.std(init_den, keepdims=True)).float()
    init_den = torch.tensor(init_den).float()
    init_den = init_den.cuda()
    init_den_mean = init_den_mean.cuda()
    init_den_std = init_den_std.cuda()
    init_den_normalization = Normalization(mean_val=init_den_mean, std_val=init_den_std)
    init_den = init_den_normalization.normalize(init_den)  # (3400,1,1200)tensor

    init_data = torch.cat([init_vs, init_vp, init_den], dim=1)  # (3400,3,1200)tensor

    seismic_mean = torch.tensor(np.mean(seismic_data, keepdims=True)).float()
    seismic_std = torch.tensor(np.std(seismic_data, keepdims=True)).float()
    seismic_data = torch.tensor(seismic_data).float()
    seismic_data = seismic_data.cuda()
    seismic_mean = seismic_mean.cuda()
    seismic_std = seismic_std.cuda()
    seismic_normalization = Normalization(mean_val=seismic_mean, std_val=seismic_std)
    seismic_data = seismic_normalization.normalize(seismic_data)
    # 处理vs
    data_vs = data_vs[:, np.newaxis]  # (1201,1,3400)
    data_vs = np.swapaxes(data_vs, 0, 2)  # (3400,1,1201)
    data_vs = np.delete(data_vs, 0, axis=2)  # (3400,1,1200)
    vs_mean = torch.tensor(np.mean(data_vs, keepdims=True)).float()  # 均值
    vs_std = torch.tensor(np.std(data_vs, keepdims=True)).float()  # 方差
    data_vs = torch.tensor(data_vs).float()
    data_vs = data_vs.cuda()
    vs_mean = vs_mean.cuda()
    vs_std = vs_std.cuda()
    vs_normalization = Normalization(mean_val=vs_mean, std_val=vs_std)
    data_vs = vs_normalization.normalize(data_vs)
    # 处理vp
    data_vp = data_vp[:, np.newaxis]  # (1201,1,3400)
    data_vp = np.swapaxes(data_vp, 0, 2)  # (3400,1,1201)
    data_vp = np.delete(data_vp, 0, axis=2)  # (3400,1,1200)
    vp_mean = torch.tensor(np.mean(data_vp, keepdims=True)).float()  # 均值
    vp_std = torch.tensor(np.std(data_vp, keepdims=True)).float()  # 方差
    data_vp = torch.tensor(data_vp).float()
    data_vp = data_vp.cuda()
    vp_mean = vp_mean.cuda()
    vp_std = vp_std.cuda()
    vp_normalization = Normalization(mean_val=vp_mean, std_val=vp_std)
    data_vp = vp_normalization.normalize(data_vp)
    # 处理den
    data_den = data_den[:, np.newaxis]  # (1201,1,3400)
    data_den = np.swapaxes(data_den, 0, 2)  # (3400,1,1201)
    data_den = np.delete(data_den, 0, axis=2)  # (3400,1,1200)
    den_mean = torch.tensor(np.mean(data_den, keepdims=True)).float()  # 均值
    den_std = torch.tensor(np.std(data_den, keepdims=True)).float()  # 方差
    data_den = torch.tensor(data_den).float()
    data_den = data_den.cuda()
    den_mean = den_mean.cuda()
    den_std = den_std.cuda()
    den_normalization = Normalization(mean_val=den_mean, std_val=den_std)
    data_den = den_normalization.normalize(data_den)

    inverted_data = torch.cat([data_vs, data_vp, data_den], dim=1)  # (3400,3,1200)

    seismic_data_fill = torch.zeros(seismic_data.shape[0] + args.width - 1, seismic_data.shape[1],
                                    seismic_data.shape[2]).cuda()
    seismic_data_fill[int((args.width - 1) / 2):int((args.width - 1) / 2) + seismic_data.shape[0], :] = seismic_data
    # seismic_data_expand:(N, 1, arg.width, T)
    seismic_data_expand = torch.zeros(seismic_data.shape[0], seismic_data.shape[1], args.width,
                                      seismic_data.shape[2]).cuda()
    for i in range(seismic_data.shape[0]):
        seismic_data_expand[i, :] = seismic_data_fill[i:i + args.width, :].transpose(0, 1)
    # seismic_data:(N+arg.width-1, 1, arg.width+1, T)
    seismic_data = torch.cat((seismic_data_expand, seismic_data.unsqueeze(2)), dim=2)

    init_data_fill = torch.zeros(init_data.shape[0] + args.width - 1, init_data.shape[1],
                                 init_data.shape[2]).cuda()
    init_data_fill[int((args.width - 1) / 2):int((args.width - 1) / 2) + init_data.shape[0], :] = init_data

    init_data_expand = torch.zeros(init_data.shape[0], init_data.shape[1], args.width,
                                   init_data.shape[2]).cuda()
    for i in range(init_data.shape[0]):
        init_data_expand[i, :] = init_data_fill[i:i + args.width, :].transpose(0, 1)

    init_data = torch.cat((init_data_expand, init_data.unsqueeze(2)), dim=2)

    if not test:
        num_samples = seismic_data.shape[0]
        indecies = np.arange(0, num_samples)
        train_indecies = indecies[(np.linspace(0, len(indecies) - 1, args.num_train_wells)).astype(int)]

        train_data = data.Subset(data.TensorDataset(seismic_data, inverted_data, init_data), train_indecies)
        train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

        unlabeled_loader = data.DataLoader(data.TensorDataset(seismic_data, init_data), batch_size=args.batch_size,
                                           shuffle=True)
        return train_loader, unlabeled_loader
    else:
        test_loader = data.DataLoader(data.TensorDataset(seismic_data, inverted_data, init_data),
                                      batch_size=args.batch_size, shuffle=False, drop_last=False)
        return test_loader, seismic_normalization, vs_normalization, vp_normalization, den_normalization


def get_models(args):
    if args.test_checkpoint is None:
        inverse_net = inverse_model()
        forward_net = forward_model()
        optimizer = optim.Adam(list(inverse_net.parameters()) + list(forward_net.parameters()), amsgrad=True,
                               lr=0.001)
    else:
        try:
            inverse_net = torch.load("./checkpoints/" + args.test_checkpoint + "_inverse")
            forward_net = torch.load("./checkpoints/" + args.test_checkpoint + "_forward")
            optimizer = torch.load("./checkpoints/" + args.test_checkpoint + "_optimizer")
        except FileNotFoundError:
            try:
                inverse_net = torch.load(args.test_checkpoint + "_inverse")
                forward_net = torch.load(args.test_checkpoint + "_forward")
                optimizer = torch.load(args.test_checkpoint + "_optimizer")
            except:
                print("No checkpoint found at '{}'- Please specify the model for testing".format(args.test_checkpoint))
                exit()
    inverse_net.cuda()
    forward_net.cuda()

    return inverse_net, forward_net, optimizer


def train(args):
    train_loader, unlabeled_loader = get_data(args)
    inverse_net, forward_net, optimizer = get_models(args)
    inverse_net.train()
    criterion = nn.MSELoss()
    criterion_1 = nn.SmoothL1Loss(beta=0.5)

    # make a directory to save models if it doesn't exist
    if not isdir("./checkpoints"):
        os.mkdir("./checkpoints")

    loss_var = []
    print("Training the model")
    for epoch in tqdm(range(args.max_epoch)):
        train_loss = []
        for x, y, z in train_loader:
            optimizer.zero_grad()
            vs, vp, den = y[:, 0, :], y[:, 1, :], y[:, 2, :]
            vs, vp, den = vs.unsqueeze(1), vp.unsqueeze(1), den.unsqueeze(1)
            z_expand = z[:, :, 0:args.width]  # 二维反演
            z_expand = z_expand.permute(0, 1, 3, 2)
            x_expand = x[:, :, 0:args.width]  # 二维反演
            x_expand = x_expand.permute(0, 1, 3, 2)
            x = x[:, :, args.width]  # x 用于正演
            vs_pred, vp_pred, den_pred = inverse_net(x_expand, z_expand)
            x_rec = forward_net(y)

            loss_vs = 0.1 * criterion_1(vs_pred, vs) + 0.9 * criterion(vs_pred, vs)
            loss_vp = 0.1 * criterion_1(vp_pred, vp) + 0.9 * criterion(vp_pred, vp)
            loss_den = 0.1 * criterion_1(den_pred, den) + 0.9 * criterion(den_pred, den)

            property_loss = criterion(x_rec, x) + loss_vs + loss_vp + 1.4 * loss_den
            if args.beta != 0:
                try:
                    x_u, z_u = next(unlabeled)
                except:
                    unlabeled = iter(unlabeled_loader)
                    x_u, z_u = next(unlabeled)

                z_u_expand = z_u[:, :, 0:args.width]
                z_u_expand = z_u_expand.permute(0, 1, 3, 2)
                x_u_expand = x_u[:, :, 0:args.width]
                x_u_expand = x_u_expand.permute(0, 1, 3, 2)
                x_u = x_u[:, :, args.width]  # 得到x_u
                vs_u_pred, vp_u_pred, den_u_pred = inverse_net(x_u_expand, z_u_expand)
                y_u_pred = torch.cat([vs_u_pred, vp_u_pred, den_u_pred], dim=1)
                x_u_rec = forward_net(y_u_pred)

                seismic_loss = criterion(x_u_rec, x_u)
            else:
                seismic_loss = 0
            loss = args.alpha * property_loss + args.beta * seismic_loss
            loss.backward()
            optimizer.step()

            train_loss.append(loss.detach().clone())

        train_loss = torch.mean(torch.tensor(train_loss))
        loss_var.append(train_loss)
    loss_var = torch.tensor(loss_var)
    loss_var = loss_var.cpu().numpy()
    # 绘图
    plt.plot(np.arange(loss_var.shape[0]), loss_var)
    plt.title("Curve of Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.show()

    torch.save(inverse_net, "./checkpoints/{}_inverse".format(args.session_name))
    torch.save(forward_net, "./checkpoints/{}_forward".format(args.session_name))
    torch.save(optimizer, "./checkpoints/{}_optimizer".format(args.session_name))


def test(args):
    test_loader, seismic_normalization, vs_normalization, vp_normalization, den_normalization = get_data(args,
                                                                                                         test=True)

    if args.test_checkpoint is None:
        args.test_checkpoint = "./checkpoints/{}".format(args.session_name)
    inverse_net, forward_net, _ = get_models(args)

    criterion = nn.MSELoss()
    criterion_1 = nn.SmoothL1Loss(beta=0.5)
    predicted = []
    true_data = []
    test_property_corr = []
    test_property_r2 = []
    inverse_net.eval()

    print("\nTesting the model\n")

    with torch.no_grad():
        test_loss = []
        for x, y, z in test_loader:
            vs, vp, den = y[:, 0, :], y[:, 1, :], y[:, 2, :]
            vs, vp, den = vs.unsqueeze(1), vp.unsqueeze(1), den.unsqueeze(1)
            z_expand = z[:, :, 0:args.width]
            z_expand = z_expand.permute(0, 1, 3, 2)
            x_expand = x[:, :, 0:args.width]
            x_expand = x_expand.permute(0, 1, 3, 2)
            x = x[:, :, args.width]
            vs_pred, vp_pred, den_pred = inverse_net(x_expand, z_expand)
            y_pred = torch.cat([vs_pred, vp_pred, den_pred], dim=1)
            x_rec = forward_net(y_pred)
            loss_vs = 0.1 * criterion_1(vs_pred, vs) + 0.9 * criterion(vs_pred, vs)
            loss_vp = 0.1 * criterion_1(vp_pred, vp) + 0.9 * criterion(vp_pred, vp)
            loss_den = 0.1 * criterion_1(den_pred, den) + 0.9 * criterion(den_pred, den)
            loss = args.alpha * (loss_vs + loss_vp + 1.4 * loss_den) + args.beta * criterion(x_rec, x)
            test_loss.append(loss.item())

            corr, r2 = metrics(y_pred.detach(), y.detach())
            test_property_corr.append(corr)
            test_property_r2.append(r2)

            true_data.append(y)
            predicted.append(y_pred)

        property_corr = torch.mean(torch.cat(test_property_corr), dim=0).squeeze()
        property_r2 = torch.mean(torch.cat(test_property_r2), dim=0).squeeze()
        loss = torch.mean(torch.tensor(test_loss))

        property_corr_reordered = property_corr[[1, 0, 2]]
        property_r2_reordered = property_r2[[1, 0, 2]]

        print('corr (vp, vs, den):\n', property_corr_reordered)
        print('r2 (vp, vs, den):\n', property_r2_reordered)
        print('loss:\n', loss)


        predicted = torch.cat(predicted, dim=0)  # 得到预测(3400,3,1200) tensor
        true_data = torch.cat(true_data, dim=0)  # 得到真实(3400,3,1200) tensor

        predicted = torch.chunk(predicted, 3, dim=1)
        predicted_vs = predicted[0]  # 得到预测的vs:[3400,1,1200]
        predicted_vp = predicted[1]  # 得到预测的vp:[3400,1,1200]
        predicted_den = predicted[2]  # 得到预测的den:[3400,1,1200]

        true_data = torch.chunk(true_data, 3, dim=1)
        true_vs = true_data[0]  # [3400,1,1200]
        true_vp = true_data[1]  # [3400,1,1200]
        true_den = true_data[2]  # [3400,1,1200]

        predicted_vpp = predicted_vp.cpu().numpy()
        true_vpp = true_vp.cpu().numpy()
        print('vp的MSE: {:0.4f}'.format(np.sum((predicted_vpp - true_vpp).ravel() ** 2) / predicted_vpp.size))

        predicted_vss = predicted_vs.cpu().numpy()
        true_vss = true_vs.cpu().numpy()
        print('vs的MSE: {:0.4f}'.format(np.sum((predicted_vss - true_vss).ravel() ** 2) / predicted_vss.size))

        predicted_denn = predicted_den.cpu().numpy()
        true_denn = true_den.cpu().numpy()
        print('den的MSE: {:0.4f}'.format(np.sum((predicted_denn - true_denn).ravel() ** 2) / predicted_denn.size))
        # =========================================================================

        # 规范化 （解除归一化)
        predicted_vs = vs_normalization.unnormalize(predicted_vs)
        true_vs = vs_normalization.unnormalize(true_vs)
        predicted_vp = vp_normalization.unnormalize(predicted_vp)
        true_vp = vp_normalization.unnormalize(true_vp)
        predicted_den = den_normalization.unnormalize(predicted_den)
        true_den = den_normalization.unnormalize(true_den)

        # 加载到CPU上 并转化成numpy
        predicted_vs = predicted_vs.cpu()
        true_vs = true_vs.cpu()
        predicted_vs = predicted_vs.numpy()
        true_vs = true_vs.numpy()

        predicted_vp = predicted_vp.cpu()
        true_vp = true_vp.cpu()
        predicted_vp = predicted_vp.numpy()
        true_vp = true_vp.numpy()

        predicted_den = predicted_den.cpu()
        true_den = true_den.cpu()
        predicted_den = predicted_den.numpy()
        true_den = true_den.numpy()

        init_vs = true_vs.reshape(3400, 1200)
        init_vs = cv2.GaussianBlur(init_vs, (k_size, k_size), 0)

        init_vp = true_vp.reshape(3400, 1200)
        init_vp = cv2.GaussianBlur(init_vp, (k_size, k_size), 0)

        init_den = true_den.reshape(3400, 1200)
        init_den = cv2.GaussianBlur(init_den, (k_size, k_size), 0)

        # np.save('./pred_data/predicted_vs_noisy', predicted_vs)
        # np.save('./pred_data/predicted_vp_noisy', predicted_vp)
        # np.save('./pred_data/predicted_den_noisy', predicted_den)

        # -------vp----------------------------------------------------------------------------------------------------------------------
        plt.subplot(1, 1, 1)
        x1 = true_vp[200][0]
        x2 = predicted_vp[200][0]
        x3 = init_vp[200]
        plt.plot(x1, color='black', label='true vp', linewidth=1)
        plt.plot(x2, color='red', label='predicted vp', linewidth=1)
        plt.plot(x3, color='green', label='initial vp', linewidth=1)
        plt.legend()
        plt.xlabel('Sampling points')
        plt.ylabel('P-velocity')
        plt.title('No.200 CDP')
        plt.show()

        error_map = abs(true_vp[:, 0].T - predicted_vp[:, 0].T)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        ##########################
        # 1. 真实 vp
        ##########################
        im1 = axs[0].imshow(true_vp[:, 0].T, vmin=true_vp.min(), vmax=true_vp.max(), cmap="viridis")
        axs[0].set_aspect(1.55)
        axs[0].set_title("True Vp", fontsize=14)
        axs[0].set_xlabel("CDP", fontsize=12)
        axs[0].set_ylabel("Sampling points", fontsize=12)
        cb1 = fig.colorbar(im1, ax=axs[0], fraction=0.0256, pad=0.04)
        cb1.ax.tick_params(labelsize=12)
        cb1.set_label("P-velocity (m/s)", fontsize=12)

        ##########################
        # 2. 预测 vp
        ##########################
        im2 = axs[1].imshow(predicted_vp[:, 0].T, vmin=true_vp.min(), vmax=true_vp.max(), cmap="viridis")
        axs[1].set_aspect(1.55)
        axs[1].set_title("Predicted Vp", fontsize=14)
        axs[1].set_xlabel("CDP", fontsize=12)
        axs[1].set_ylabel("Sampling points", fontsize=12)
        cb2 = fig.colorbar(im2, ax=axs[1], fraction=0.0256, pad=0.04)
        cb2.ax.tick_params(labelsize=12)
        cb2.set_label("P-velocity (m/s)", fontsize=12)

        ##########################
        # 3. 误差图（Absolute Error）
        ##########################
        im3 = axs[2].imshow(error_map, vmin=0, vmax=2000, cmap="hot_r")
        axs[2].set_aspect(1.55)
        axs[2].set_title("Absolute Error", fontsize=14)
        axs[2].set_xlabel("CDP", fontsize=12)
        axs[2].set_ylabel("Sampling points", fontsize=12)
        cb3 = fig.colorbar(im3, ax=axs[2], fraction=0.0256, pad=0.04)
        cb3.ax.tick_params(labelsize=12)
        cb3.set_label("Absolute Error (m/s)", fontsize=12)

        plt.tight_layout()
        plt.show()

        # ---------vs------------------------------------------------------------------------------------------------------------------------------------------------------
        plt.subplot(1, 1, 1)
        x1 = true_vs[200][0]
        x2 = predicted_vs[200][0]
        x3 = init_vs[200]
        plt.plot(x1, color='black', label='true vs', linewidth=1)
        plt.plot(x2, color='red', label='predicted vs', linewidth=1)
        plt.plot(x3, color='green', label='initial vs', linewidth=1)
        plt.legend()
        plt.xlabel('Sampling points')
        plt.ylabel('S-velocity')
        plt.title('No.200 CDP')
        plt.show()

        error_map = abs(true_vs[:, 0].T - predicted_vs[:, 0].T)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        ##########################
        # 1. 真实 vs
        ##########################
        im1 = axs[0].imshow(true_vs[:, 0].T, vmin=true_vs.min(), vmax=true_vs.max(), cmap="viridis")
        axs[0].set_aspect(1.55)
        axs[0].set_title("True Vs", fontsize=14)
        axs[0].set_xlabel("CDP", fontsize=12)
        axs[0].set_ylabel("Sampling points", fontsize=12)
        cb1 = fig.colorbar(im1, ax=axs[0], fraction=0.0256, pad=0.04)
        cb1.ax.tick_params(labelsize=12)
        cb1.set_label("S-velocity (m/s)", fontsize=12)

        ##########################
        # 2. 预测 vs
        ##########################
        im2 = axs[1].imshow(predicted_vs[:, 0].T, vmin=true_vs.min(), vmax=true_vs.max(), cmap="viridis")
        axs[1].set_aspect(1.55)
        axs[1].set_title("Predicted Vs", fontsize=14)
        axs[1].set_xlabel("CDP", fontsize=12)
        axs[1].set_ylabel("Sampling points", fontsize=12)
        cb2 = fig.colorbar(im2, ax=axs[1], fraction=0.0256, pad=0.04)
        cb2.ax.tick_params(labelsize=12)
        cb2.set_label("S-velocity (m/s)", fontsize=12)

        ##########################
        # 3. 误差图（Absolute Error）
        ##########################
        im3 = axs[2].imshow(error_map, vmin=0, vmax=2000, cmap="hot_r")
        axs[2].set_aspect(1.55)
        axs[2].set_title("Absolute Error", fontsize=14)
        axs[2].set_xlabel("CDP", fontsize=12)
        axs[2].set_ylabel("Sampling points", fontsize=12)
        cb3 = fig.colorbar(im3, ax=axs[2], fraction=0.0256, pad=0.04)
        cb3.ax.tick_params(labelsize=12)
        cb3.set_label("Absolute Error (m/s)", fontsize=12)

        plt.tight_layout()
        plt.show()

        # -------den----------------------------------------------------------------------------------------------------------------------
        plt.subplot(1, 1, 1)
        x1 = true_den[200][0]
        x2 = predicted_den[200][0]
        x3 = init_den[200]
        plt.plot(x1, color='black', label='true density', linewidth=1)
        plt.plot(x2, color='red', label='predicted density', linewidth=1)
        plt.plot(x3, color='green', label='initial density', linewidth=1)
        plt.legend()
        plt.xlabel('Sampling points')
        plt.ylabel('Density')
        plt.title('No.200 CDP')
        plt.show()

        error_map = abs(true_den[:, 0].T - predicted_den[:, 0].T)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        ##########################
        # 1. 真实 den
        ##########################
        im1 = axs[0].imshow(true_den[:, 0].T, vmin=true_den.min(), vmax=true_den.max(), cmap="viridis")
        axs[0].set_aspect(1.55)
        axs[0].set_title("True Den", fontsize=14)
        axs[0].set_xlabel("CDP", fontsize=12)
        axs[0].set_ylabel("Sampling points", fontsize=12)
        cb1 = fig.colorbar(im1, ax=axs[0], fraction=0.0256, pad=0.04)
        cb1.ax.tick_params(labelsize=12)
        cb1.set_label("Density (g/cm³)", fontsize=12)

        ##########################
        # 2. 预测 den
        ##########################
        im2 = axs[1].imshow(predicted_den[:, 0].T, vmin=true_den.min(), vmax=true_den.max(), cmap="viridis")
        axs[1].set_aspect(1.55)
        axs[1].set_title("Predicted Den", fontsize=14)
        axs[1].set_xlabel("CDP", fontsize=12)
        axs[1].set_ylabel("Sampling points", fontsize=12)
        cb2 = fig.colorbar(im2, ax=axs[1], fraction=0.0256, pad=0.04)
        cb2.ax.tick_params(labelsize=12)
        cb2.set_label("Density (g/cm³)", fontsize=12)

        ##########################
        # 3. 误差图（Absolute Error）
        ##########################
        im3 = axs[2].imshow(error_map, vmin=0, vmax=1.0, cmap="hot_r")
        axs[2].set_aspect(1.55)
        axs[2].set_title("Absolute Error", fontsize=14)
        axs[2].set_xlabel("CDP", fontsize=12)
        axs[2].set_ylabel("Sampling points", fontsize=12)
        cb3 = fig.colorbar(im3, ax=axs[2], fraction=0.0256, pad=0.04)
        cb3.ax.tick_params(labelsize=12)
        cb3.set_label("Absolute Error (g/cm³)", fontsize=12)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # arguments and parameters代码中所有的在arg中的参数都在这里设置
    parser = argparse.ArgumentParser()
    parser.add_argument('-width', type=int, default=3,
                        help="Number of seismic traces in expanding data to be used for training. It must be odd奇数")
    parser.add_argument('-num_train_wells', type=int, default=12,
                        help="Number of AI traces from the model to be used for training")
    parser.add_argument('-max_epoch', type=int, default=1000, help="maximum number of training epochs")
    parser.add_argument('-batch_size', type=int, default=12, help="Batch size for training")
    parser.add_argument('-alpha', type=float, default=1, help="weight of property loss term")
    parser.add_argument('-beta', type=float, default=0.2, help="weight of seismic loss term")
    parser.add_argument('-test_checkpoint', type=str, action="store", default=None,
                        help="path to model to test on. When this flag is used, no training is performed")
    parser.add_argument('-session_name', type=str, action="store", default=datetime.now().strftime('%b%d_%H%M%S'),
                        help="name of the session to be ised in saving the model")
    args = parser.parse_args()

    if args.test_checkpoint is not None:
        test(args)
    else:
        train(args)
        test(args)
