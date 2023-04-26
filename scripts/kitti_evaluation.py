import sys
sys.path.append('.')

from tqdm import tqdm
import os
import numpy as np
import cv2
import argparse
import torch

from lietorch import SE3
import raft3d.projective_ops as pops

from utils import show_image, normalize_image
from data_readers.kitti import KITTIEval, KITTI
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from glob import glob
from data_readers.frame_utils import *


def display(img, tau, phi):
    """ display se3 fields """
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.imshow(img[:, :, ::-1] / 255.0)

    tau_img = np.clip(tau, -0.1, 0.1)
    tau_img = (tau_img + 0.1) / 0.2

    phi_img = np.clip(phi, -0.1, 0.1)
    phi_img = (phi_img + 0.1) / 0.2

    ax2.imshow(tau_img)
    ax3.imshow(phi_img)
    plt.show()


def prepare_images_and_depths(image1, image2, depth1, depth2, depth_scale=1.0):
    """ padding, normalization, and scaling """
    
    ht, wd = image1.shape[-2:]
    pad_h = (-ht) % 8
    pad_w = (-wd) % 8

    image1 = F.pad(image1, [0,pad_w,0,pad_h], mode='replicate')
    image2 = F.pad(image2, [0,pad_w,0,pad_h], mode='replicate')
    depth1 = F.pad(depth1[:,None], [0,pad_w,0,pad_h], mode='replicate')[:,0]
    depth2 = F.pad(depth2[:,None], [0,pad_w,0,pad_h], mode='replicate')[:,0]

    depth1 = (depth_scale * depth1).float()
    depth2 = (depth_scale * depth2).float()
    image1 = normalize_image(image1.float())
    image2 = normalize_image(image2.float())

    depth1 = depth1.float()
    depth2 = depth2.float()

    return image1, image2, depth1, depth2, (pad_w, pad_h)


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    # perturbed_image = torch.clamp(perturbed_image, 0, 255)
    return perturbed_image


# @torch.no_grad()
def make_kitti_submission(model):
    if args.attack_type != 'None':
        torch.set_grad_enabled(True) 
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 1, 'drop_last': False}
    train_dataset = KITTI()
    train_loader = DataLoader(train_dataset, **loader_args)

    count_all, count_sampled = 0, 0
    metrics_all = {'epe2d': 0.0, 'epe3d': 0.0, '1px': 0.0, '5cm': 0.0, '10cm': 0.0}


    DEPTH_SCALE = .1

    for i_batch, data_blob in enumerate(tqdm(train_loader)):
        image1, image2, depth1, depth2, flow, _, intrinsics = \
            [data_item.cuda() for data_item in data_blob]
        
        if args.attack_type != 'None':
            image1.requires_grad = True # for attack

        ht, wd = image1.shape[2:]
        image1_t, image2_t, depth1_t, depth2_t, padding = \
            prepare_images_and_depths(image1, image2, depth1, depth2)

        # depth1_t.requires_grad = True # for attack


        Ts = model(image1_t, image2_t, depth1_t, depth2_t, intrinsics, iters=16)
        # print(torch.min(image1), torch.max(image1))
        # print(depth1.shape, torch.max(depth1), torch.min(depth1))
        # break
        
        # tau_phi = Ts.log()

        # uncomment to diplay motion field
        # tau, phi = Ts.log().split([3,3], dim=-1)
        # tau = tau[0].cpu().numpy()
        # phi = phi[0].cpu().numpy()
        # display(img1, tau, phi)

        # compute optical flow
        flow2d_est, flow3d_est, _ = pops.induced_flow(Ts, depth1_t, intrinsics)
        
        flow2d_est = flow2d_est[0, :ht, :wd, :2]
        flow3d_est = flow3d_est[:, :ht, :wd] / DEPTH_SCALE

        # start attack
        if args.attack_type != 'None':
            if args.attack_type == 'FGSM':
                epsilon = args.epsilon
                pgd_iters = 1
            else:
                epsilon = args.epsilon / args.iters
                pgd_iters = args.iters
        
            for iter in range(pgd_iters):
                epe3d = torch.sum((flow3d_est - flow)**2, -1).sqrt()
                model.zero_grad()
                epe3d.mean().backward()
                # data_grad = depth1_t.grad.data
                # depth1_t.data = fgsm_attack(depth1_t, 2, data_grad)
                data_grad = image1.grad.data
                if args.channel == -1:
                    image1.data = fgsm_attack(image1, epsilon, data_grad)
                else:
                    image1.data[:, args.channel, :, :] = fgsm_attack(image1, epsilon, data_grad)[:, args.channel, :, :]
                image1_t, image2_t, depth1_t, depth2_t, padding = prepare_images_and_depths(image1, image2, depth1, depth2)

                Ts = model(image1_t, image2_t, depth1_t, depth2_t, intrinsics, iters=16)
                flow2d_est, flow3d_est, _ = pops.induced_flow(Ts, depth1_t, intrinsics)
                flow2d_est = flow2d_est[0, :ht, :wd, :2]
                flow3d_est = flow3d_est[:, :ht, :wd] / DEPTH_SCALE
        # end attack
        
        epe2d = torch.sum((flow2d_est - flow[0, :, :, :2])**2, -1).sqrt()
        epe3d = torch.sum((flow3d_est - flow)**2, -1).sqrt()
        epe2d_all = epe2d.reshape(-1).double().cpu().detach().numpy()
        epe3d_all = epe3d.reshape(-1).double().cpu().detach().numpy()
        
        count_all += epe2d_all.shape[0]
        metrics_all['epe2d'] += epe2d_all.sum()
        metrics_all['epe3d'] += epe3d_all.sum()
        metrics_all['1px'] += np.count_nonzero(epe2d_all < 1.0)
        metrics_all['5cm'] += np.count_nonzero(epe3d_all < .05)
        metrics_all['10cm'] += np.count_nonzero(epe3d_all < .10)
    
    # Average results over all valid pixels
    print("all...")
    for key in metrics_all:
        print(key, metrics_all[key] / count_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path the model weights')
    parser.add_argument('--network', default='raft3d.raft3d', help='network architecture')
    parser.add_argument('--radius', type=int, default=32)
    parser.add_argument('--attack_type', help='Attack type options: None, FGSM, PGD', type=str, default='PGD')
    parser.add_argument('--iters', help='Number of iters for PGD?', type=int, default=10)
    parser.add_argument('--epsilon', help='epsilon?', type=int, default=10)
    parser.add_argument('--channel', help='Color channel options: 0, 1, 2, -1 (all)', type=int, default=-1) 
    args = parser.parse_args()

    import importlib
    RAFT3D = importlib.import_module(args.network).RAFT3D

    model = torch.nn.DataParallel(RAFT3D(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    if not os.path.isdir('kitti_submission'):
        os.mkdir('kitti_submission')
        os.mkdir('kitti_submission/disp_0')
        os.mkdir('kitti_submission/disp_1')
        os.mkdir('kitti_submission/flow')

    make_kitti_submission(model)
