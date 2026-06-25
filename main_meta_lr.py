import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--spokes', type=int, metavar='', required=True)
parser.add_argument('-g', '--gpu', type=int, metavar='', required=True)
parser.add_argument('-t', '--tv_weight', type=float, metavar='', required=False, default=0.02)
parser.add_argument('-l', '--lr_weight', type=float, metavar='', required=False, default=0.0002)
parser.add_argument('-st', '--stv_weight', type=float, metavar='', required=False, default=0) # Just in case
parser.add_argument('-n', '--neuron', type=int, metavar='', required=False, default=128)
parser.add_argument('-ly', '--layers', type=int, metavar='', required=False, default=5)
parser.add_argument('-hs', '--log2_hashmap_size', type=int, metavar='', required=False, default=24)
parser.add_argument('-ls', '--per_level_scale', type=float, metavar='', required=False, default=2.0)
parser.add_argument('-e', '--epochs', type=int, metavar='', required=False, default=1600)
parser.add_argument('-m', '--mask', action='store_true', required=False)
parser.add_argument('-r', '--relL2', action='store_true', required=False)
# Meta-learning specific
parser.add_argument('--meta_epochs', type=int, default=100, help='Number of meta-training iterations')
parser.add_argument('--meta_lr', type=float, default=0.1, help='Meta learning rate (Reptile step size)')
parser.add_argument('--inner_steps', type=int, default=5, help='Inner adaptation steps per task')
parser.add_argument('--tasks_per_meta', type=int, default=4, help='Number of tasks sampled per meta-iteration')
parser.add_argument('--task_frames', type=int, default=1, help='Number of frames per task (support set size)')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import numpy as np
import torch
import datetime
import h5py
from tqdm import tqdm
import copy

from torch.utils.tensorboard import SummaryWriter
from inr.utils import coil_combine, path_checker, visual_mag, visual_err_mag, gen_traj, NUFFT
from inr.model import INR
from scipy import io

params = {
    'n_levels': 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": args.log2_hashmap_size,
    "base_resolution": 16,
    "per_level_scale": args.per_level_scale,
    'lr': 0.001,
    "n_neurons": args.neuron,
    "n_hidden_layers": args.layers,
    "tv_weight": args.tv_weight,
    "lr_weight": args.lr_weight,
    "stv_weight": args.stv_weight,
    "epochs": args.epochs, 
    "mask": args.mask,
    "relL2": args.relL2
}
print(params)

# Important Constants
GA = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))  # GoldenAngle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 1e-3
summary_epoch = 50
spoke_num = args.spokes
epoch = params['epochs']
relL2_eps = 1e-4

log_path = './log_cmr_meta/spoke{}_{}'.format(spoke_num, str(datetime.datetime.now().strftime('%y%m%d_%H%M%S')))
path_checker(log_path)
writer = SummaryWriter(log_path)

# Import and Preprocess Data
mat_path = './test_cardiac.mat'
with h5py.File(mat_path, 'r') as f:
    img = f['img'][:]
    smap = f['smap'][:]
img = torch.as_tensor(img).to(device)
smap = torch.as_tensor(smap).to(device)
frames = img.shape[0]
coil_num = img.shape[1]
grid_size = img.shape[-1]
spoke_length = grid_size * 2
img_gt = coil_combine(img, smap)
scale_factor = torch.abs(img_gt).max()
img_gt /= scale_factor # Normalization
# Full k-space trajectory and density compensation (shared)
ktraj = gen_traj(GA, spoke_length, frames * spoke_num).reshape(2, frames, -1).transpose(1, 0)
dcomp = torch.abs(torch.linspace(-1, 1, spoke_length)).repeat([spoke_num, 1]).to(device)
# Full NUFFT and k-space data
full_nufft_op = NUFFT(ktraj, dcomp, smap)
kdata = full_nufft_op.forward(img_gt).reshape([frames, coil_num, spoke_num, spoke_length])

# Initialize meta model
meta_inr = INR(full_nufft_op, params, lr, relL2_eps)
# Use meta parameters on device
meta_inr.to(device) if hasattr(meta_inr, 'to') else None

# Helper: build task-specific NUFFT and kdata given frame indices

def build_task(nufft_ktraj, kdata_full, idx):
    # idx: list or 1D array of frame indices
    task_ktraj = nufft_ktraj[idx]
    task_kdata = kdata_full[idx]
    task_nufft = NUFFT(task_ktraj, dcomp, smap)
    return task_nufft, task_kdata

# Reptile-style meta-training loop
meta_epochs = args.meta_epochs
meta_lr = args.meta_lr
inner_steps = args.inner_steps
tasks_per_meta = args.tasks_per_meta
task_frames = min(args.task_frames, frames)

best_psnr = 0.0
for me in range(meta_epochs):
    # Sample tasks and accumulate adapted weights
    meta_state = {k: v.clone().detach() for k, v in meta_inr.state_dict().items()}
    for t in range(tasks_per_meta):
        # sample task frames without replacement
        idx = np.random.choice(frames, task_frames, replace=False)
        task_nufft, task_kdata = build_task(ktraj, kdata, idx)
        # create a fresh INR for adaptation and load meta weights
        adapt_inr = INR(task_nufft, params, lr, relL2_eps)
        adapt_inr.load_state_dict(meta_inr.state_dict())
        adapt_inr.to(device) if hasattr(adapt_inr, 'to') else None
        # build positions for task
        pos_task = adapt_inr.build_pos(task_nufft.grid_size, task_nufft.frame_num)
        # Inner adaptation
        for istep in range(inner_steps):
            adapt_inr.train(pos_task, task_kdata, istep)
        # Get adapted weights
        adapted_state = adapt_inr.state_dict()
        # Reptile meta-update: move meta_state toward adapted_state
        with torch.no_grad():
            for k in meta_state.keys():
                meta_state[k] = meta_state[k] + meta_lr * (adapted_state[k].to(meta_state[k].device) - meta_state[k])
    # Load updated meta parameters
    meta_inr.load_state_dict(meta_state)

    # Optional evaluation on full data every few meta-iterations
    if (me + 1) % (max(1, summary_epoch // 10)) == 0 or me == meta_epochs - 1:
        with torch.no_grad():
            pos_full = meta_inr.build_pos(grid_size, frames)
            intensity, psnr_tmp, ssim_tmp = meta_inr.infer(pos_full, img_gt, smap)
        io.savemat(log_path + '/meta_proposed_{}.mat'.format(me+1), {'img_proposed': intensity.cpu().numpy()})
        visual_mag(intensity, log_path + '/meta_proposed_{}_{}_abs_{}.png'.format(spoke_num, frames, me+1))
        visual_err_mag(intensity, img_gt, log_path + '/meta_proposed_{}_{}_abs_err_{}.png'.format(spoke_num, frames, me+1))
        writer.add_scalar('meta_psnr', psnr_tmp, me + 1)
        writer.add_scalar('meta_ssim', ssim_tmp, me + 1)
        print('[MetaIter {}/{}] PSNR: {:.4f} SSIM: {:.4f}'.format(me+1, meta_epochs, psnr_tmp, ssim_tmp))
        if psnr_tmp > best_psnr:
            best_psnr = psnr_tmp

# Final summary
print('Best Meta PSNR: {:.4f}'.format(best_psnr))
print('Meta-training finished')
