import sys
sys.path.append('.')

from tqdm import tqdm
import numpy as np
import cv2
import argparse
import torch

from lietorch import SE3
import raft3d.projective_ops as pops

from utils import show_image, normalize_image
from data_readers.sceneflow import FlyingThingsTest
import torch.nn.functional as F
from torch.utils.data import DataLoader

from glob import glob
from data_readers.frame_utils import *

# scale input depth maps (scaling is undone before evaluation)
DEPTH_SCALE = 0.2

# exclude pixels with depth > 250
MAX_DEPTH = 250

# exclude extermely fast moving pixels
MAX_FLOW = 250


def prepare_images_and_depths(image1, image2, depth1, depth2, depth_scale=0.2):
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


# @torch.no_grad()
def test_sceneflow(model):
    if args.attack_type != 'None':
        torch.set_grad_enabled(True) 
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 4, 'drop_last': False}
    train_dataset = FlyingThingsTest()
    train_loader = DataLoader(train_dataset, **loader_args)

    count_all, count_sampled = 0, 0
    metrics_all = {'epe2d': 0.0, 'epe3d': 0.0, '1px': 0.0, '5cm': 0.0, '10cm': 0.0}
    metrics_flownet3d = {'epe3d': 0.0, '5cm': 0.0, '10cm': 0.0}

    for i_batch, test_data_blob in enumerate(tqdm(train_loader)):
        image1, image2, depth1, depth2, flow2d, flow3d, intrinsics, index = \
            [data_item.cuda() for data_item in test_data_blob]

        if args.attack_type != 'None':
            image1.requires_grad = True # for attack

        mag = torch.sum(flow2d**2, dim=-1).sqrt()
        valid = (mag.reshape(-1) < MAX_FLOW) & (depth1.reshape(-1) < MAX_DEPTH)

        # pad and normalize images
        image1, image2, depth1, depth2, padding = \
            prepare_images_and_depths(image1, image2, depth1, depth2, DEPTH_SCALE)

        # run the model
        Ts = model(image1, image2, depth1, depth2, intrinsics, iters=16)

        # use transformation field to extract 2D and 3D flow
        flow2d_est, flow3d_est, _ = pops.induced_flow(Ts, depth1, intrinsics)
        
        # unpad the flow fields / undo depth scaling
        flow2d_est = flow2d_est[:, :-4, :, :2]
        flow3d_est = flow3d_est[:, :-4] / DEPTH_SCALE


        # start attack
        if args.attack_type != 'None':
            ori = image1.data

            if args.attack_type == "RAND":
                epsilon = args.epsilon
                shape = image1.shape
                delta = (np.random.rand(np.product(shape)).reshape(shape) - 0.5) * 2 * epsilon
                image1.data = ori + torch.from_numpy(delta).type(torch.float).cuda()
                image1.data = torch.clamp(image1.data, 0.0, 255.0)
                image1_t, image2_t, depth1_t, depth2_t, padding = prepare_images_and_depths(image1, image2, depth1, depth2)
                Ts = model(image1_t, image2_t, depth1_t, depth2_t, intrinsics, iters=16)
                flow2d_est, flow3d_est, _ = pops.induced_flow(Ts, depth1_t, intrinsics)
                flow2d_est = flow2d_est[0, :ht, :wd, :2]
                flow3d_est = flow3d_est[:, :ht, :wd] / DEPTH_SCALE                
                pgd_iters = 0
            elif args.attack_type == 'FGSM':
                epsilon = args.epsilon
                pgd_iters = 1
            else:
                epsilon = 2.5 * args.epsilon / args.iters
                pgd_iters = args.iters

            for itr in range(pgd_iters):
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
                if args.attack_type == 'PGD':
                    image1.data = ori + torch.clamp(image1.data - ori, -args.epsilon, args.epsilon)
                image1_t, image2_t, depth1_t, depth2_t, padding = prepare_images_and_depths(image1, image2, depth1, depth2, DEPTH_SCALE)

                Ts = model(image1_t, image2_t, depth1_t, depth2_t, intrinsics, iters=16)
                flow2d_est, flow3d_est, _ = pops.induced_flow(Ts, depth1_t, intrinsics)
                flow2d_est = flow2d_est[:, :-4, :, :2]
                flow3d_est = flow3d_est[:, :-4] / DEPTH_SCALE
        # end attack
       

        epe2d = torch.sum((flow2d_est - flow2d)**2, -1).sqrt()
        epe3d = torch.sum((flow3d_est - flow3d)**2, -1).sqrt()

        # our evaluation (use all valid pixels)
        epe2d_all = epe2d.reshape(-1)[valid].double().cpu().numpy()
        epe3d_all = epe3d.reshape(-1)[valid].double().cpu().numpy()
        
        count_all += epe2d_all.shape[0]
        metrics_all['epe2d'] += epe2d_all.sum()
        metrics_all['epe3d'] += epe3d_all.sum()
        metrics_all['1px'] += np.count_nonzero(epe2d_all < 1.0)
        metrics_all['5cm'] += np.count_nonzero(epe3d_all < .05)
        metrics_all['10cm'] += np.count_nonzero(epe3d_all < .10)

        # FlowNet3D evaluation (only use sampled non-occ pixels)
        # epe3d = epe3d[0,index[0,0],index[0,1]]
        # epe2d = epe2d[0,index[0,0],index[0,1]]
        epe3d = epe3d[0]
        epe2d = epe2d[0]

        epe2d_sampled = epe2d.reshape(-1).double().cpu().numpy()
        epe3d_sampled = epe3d.reshape(-1).double().cpu().numpy()
        
        count_sampled += epe2d_sampled.shape[0]
        metrics_flownet3d['epe3d'] += epe3d_sampled.mean()
        metrics_flownet3d['5cm'] += (epe3d_sampled < .05).astype(np.float).mean()
        metrics_flownet3d['10cm'] += (epe3d_sampled < .10).astype(np.float).mean()

    # Average results over all valid pixels
    print("all...")
    for key in metrics_all:
        print(key, metrics_all[key] / count_all)

    # FlowNet3D evaluation methodology
    print("non-occ (FlowNet3D Evaluation)...")
    for key in metrics_flownet3d:
        print(key, metrics_flownet3d[key] / (i_batch + 1))


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
    model.load_state_dict(torch.load(args.model), strict=False)

    model.cuda()
    model.eval()

    test_sceneflow(model)
