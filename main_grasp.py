import os
import argparse
import time
import datetime
import numpy as np
import torch
import h5py
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from utils import coil_combine, path_checker, visual_mag, visual_err_mag, gen_traj, NUFFT
from scipy import io
from CineDataset import CineDataset, CMRxReconToINRDataset
from inr.utils import metrics_extended 
from utils import TVLoss, coil_combine, metrics, RelL2Loss

# --- Arguments for GRASP ---
parser = argparse.ArgumentParser(description="GRASP MRI Reconstruction")
parser.add_argument('-s', '--spokes', type=int, metavar='', required=True, help="Number of spokes per frame")
parser.add_argument('-g', '--gpu', type=int, metavar='', required=True, help="GPU ID")
parser.add_argument('-t', '--tv_weight', type=float, metavar='', required=False, default=0.01, help="Temporal TV penalty weight")
parser.add_argument('-lr', '--learning_rate', type=float, metavar='', required=False, default=0.01, help="Adam learning rate")
parser.add_argument('-i', '--iterations', type=int, metavar='', required=False, default=1000, help="Optimization iterations")
parser.add_argument('-r', '--relL2', action='store_true', required=False)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

params = {
    "tv_weight": args.tv_weight,
    "lr": args.learning_rate,
    "iterations": args.iterations,
    "spokes": args.spokes,
    "relL2": args.relL2
}
print("GRASP Parameters:", params)

# Important Constants
GA = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))  # GoldenAngle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
summary_epoch = 50
spoke_num = args.spokes
iterations = params['iterations']
relL2_eps = 1e-4

base_path = r"/synology-data/users/naamagav/CMRxRec_for_project"
log_path = os.path.join(base_path, 'log_cmr', 'GRASP_spoke{}_{}'.format(spoke_num, str(datetime.datetime.now().strftime('%y%m%d_%H%M%S'))))
path_checker(log_path)
writer = SummaryWriter(log_path)
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
img_gt = coil_combine(img, smap)
scale_factor = torch.abs(img_gt).max()
img_gt /= scale_factor # Normalization

# Generate K-Space Trajectory and Data
ktraj = gen_traj(GA, spoke_length, frames * spoke_num).reshape(2, frames, -1).transpose(1, 0)
dcomp = torch.abs(torch.linspace(-1, 1, spoke_length)).repeat([spoke_num, 1]).to(device)
nufft_op = NUFFT(ktraj, dcomp, smap)
kdata = nufft_op.forward(img_gt).reshape([frames, coil_num, spoke_num, spoke_length])

# --- Build GRASP Optimization Variable ---
# Initialize the image series as a complex tensor (optimizable parameter)
# Starting with zeros
# recon_img = torch.zeros((frames, 1, grid_size, grid_size), dtype=torch.complex64, device=device, requires_grad=True)
print("Calculating Adjoint NUFFT for initialization...")
with torch.no_grad():
    # Get the zero-filled reconstruction to use as a starting point
    initial_guess = nufft_op.adjoint(kdata)
    
    # Ensure it has the [frames, 1, H, W] shape
    if initial_guess.dim() == 3:
        initial_guess = initial_guess.unsqueeze(1)

# Initialize the optimizable parameter with the adjoint image
recon_img = initial_guess.clone().detach().requires_grad_(True)
optimizer = torch.optim.Adam([recon_img], lr=params['lr'])
DC_loss = RelL2Loss(params['relL2'], eps=relL2_eps)
TV_loss = TVLoss()

psnr = 0.0
ssim = 0.0
time_usage = 0.0
best_epoch = 0
best_intensity = None
epoch_loop = tqdm(range(iterations), total=iterations, leave=True)

for i in epoch_loop:
    start_time = time.time()
    optimizer.zero_grad()
    intensity = recon_img

    kdata_pred = nufft_op.forward(intensity).reshape([frames, coil_num, spoke_num, spoke_length])
    dc_loss_val = DC_loss(kdata_pred, kdata).mean()


    # 2. Temporal Total Variation (TV) Loss: || T(x) ||_1
    # Absolute difference along the temporal dimension (frames)
    tv_loss_val = (TV_loss(intensity.real) + TV_loss(intensity.imag)) / torch.abs(intensity.detach()).max()

    # Total Objective
    loss = dc_loss_val + params['tv_weight'] * tv_loss_val
    loss.backward()
    optimizer.step()

    delta_time = time.time() - start_time
    time_usage += delta_time

    # Update Progress Bar
    epoch_loop.set_description("[GRASP Opt] [Iter:{}/{}]".format(i + 1, iterations))
    epoch_loop.set_postfix(total_loss=loss.item(), dc_loss=dc_loss_val.item(), tv_loss=tv_loss_val.item())
    
    writer.add_scalar('loss_total', loss.item(), i + 1)
    writer.add_scalar('loss_dc', dc_loss_val.item(), i + 1)
    writer.add_scalar('loss_tv', tv_loss_val.item(), i + 1)

    # Inferring / Evaluation
    if (i + 1) % summary_epoch == 0:
        with torch.no_grad():

            intensity_eval = recon_img.clone().detach()   

            coil_img = intensity_eval * smap 
            combined_int = coil_combine(coil_img, smap)
            
            # Calculate metrics
            psnr_tmp, ssim_tmp = metrics(combined_int, img_gt)
        
            
            io.savemat(log_path + '/proposed_{}.mat'.format(i+1),
                        {'img_proposed': intensity_eval.cpu().numpy()})
            visual_mag(intensity_eval,
                log_path + '/proposed_{}_{}_abs_{}.png'.format(spoke_num, frames, i+1))
            visual_err_mag(intensity_eval, img_gt, log_path + '/proposed_{}_{}_abs_err_{}.png'.format(spoke_num, frames, i+1))
            
            writer.add_scalar('psnr', psnr_tmp, i + 1)
            writer.add_scalar('ssim', ssim_tmp, i + 1)
            
            if psnr_tmp > psnr:
                psnr = psnr_tmp
                ssim = ssim_tmp
                best_intensity = intensity_eval.clone().detach()
                best_epoch = i + 1

# Summary
print('--- Optimization Complete ---')
print('Best PSNR: {:.4f}'.format(psnr))
print('SSIM: {:.4f}'.format(ssim))
print('Best Iteration: {}'.format(best_epoch))
print('Time Consumption: {:.2f}s'.format(time_usage))

if best_intensity is not None:
    metrics_dict = metrics_extended(best_intensity, img_gt, time_usage, log_path + '/proposed_{}_{}_abs_err_metrics.txt'.format(spoke_num, frames))
    visual_mag(best_intensity,
    log_path + '/proposed_{}_{}_abs_{}.png'.format(spoke_num, frames, best_epoch))
    visual_err_mag(best_intensity, img_gt, log_path + '/proposed_{}_{}_abs_err_{}.png'.format(spoke_num, frames, best_epoch))
else:
    print("Warning: Optimization finished without running inference/evaluation.")