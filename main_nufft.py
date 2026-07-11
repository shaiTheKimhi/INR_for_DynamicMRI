import os
import argparse
import time
import datetime
import numpy as np
import torch
from scipy import io

# Parse only the essential arguments for the naive reconstruction
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--spokes', type=int, metavar='', required=True)
parser.add_argument('-g', '--gpu', type=int, metavar='', required=True)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

from utils import coil_combine, path_checker, visual_mag, visual_err_mag, gen_traj, NUFFT
from CineDataset import CineDataset, CMRxReconToINRDataset
from inr.utils import metrics_extended

# Important Constants
GA = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))  # GoldenAngle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
spoke_num = args.spokes

# Setup Paths
base_path = r"/synology-data/users/naamagav/CMRxRec_for_project"
log_path = os.path.join(base_path, 'log_cmr', 'naive_nufft_spoke{}_{}'.format(spoke_num, str(datetime.datetime.now().strftime('%y%m%d_%H%M%S'))))
path_checker(log_path)

dataset_path = os.path.join(base_path, 'ChallengeData_test', 'MultiCoil', 'Cine', 'TestSet', 'FullSample')
cds = CineDataset(dataset_path)

ds = CMRxReconToINRDataset(
    base_dataset=cds,
    kspace_key="kspace_full",
    z_index=2,
    input_order="nxnycnznt",
    crop_square=True,
    crop_size=204,
    return_torch=True,
)

x = ds[1]
# Test is on P002 in the testset 

img = x['img'][:]
smap = x['smap'][:]

img = torch.as_tensor(img).to(device)
smap = torch.as_tensor(smap).to(device)
frames = img.shape[0]
coil_num = img.shape[1]
grid_size = img.shape[-1]
spoke_length = grid_size * 2

# Ground Truth & Normalization
img_gt = coil_combine(img, smap)
scale_factor = torch.abs(img_gt).max()
img_gt /= scale_factor 

# Generate Trajectory & NUFFT Operator
ktraj = gen_traj(GA, spoke_length, frames * spoke_num).reshape(2, frames, -1).transpose(1, 0)
dcomp = torch.abs(torch.linspace(-1, 1, spoke_length)).repeat([spoke_num, 1]).to(device)
nufft_op = NUFFT(ktraj, dcomp, smap)

# 1. Forward Pass (Simulate multi-coil radial k-space data)
print("Simulating k-space data...")
raw_kdata = nufft_op.forward(img_gt)

# 2. Adjoint Pass (Naive NUFFT Reconstruction)
print("Running naive NUFFT reconstruction...")
start_time = time.time()

recon = nufft_op.adjoint(raw_kdata)

time_usage = time.time() - start_time
print('Time Consumption: {:.2f}s'.format(time_usage))

# Save output array
io.savemat(os.path.join(log_path, 'naive_recon.mat'), {'img_naive': recon.cpu().numpy()})

# Calculate Metrics & Save Visualizations
print("Calculating metrics and saving visuals...")
metrics_dict = metrics_extended(
    recon, 
    img_gt, 
    time_usage, 
    os.path.join(log_path, 'naive_{}_{}_abs_err_metrics.txt'.format(spoke_num, frames))
)

visual_mag(recon, os.path.join(log_path, 'naive_{}_{}_abs.png'.format(spoke_num, frames)))
visual_err_mag(recon, img_gt, os.path.join(log_path, 'naive_{}_{}_abs_err.png'.format(spoke_num, frames)))

print("Done!")