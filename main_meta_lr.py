import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--spokes', type=int, metavar='', required=False, default=10)
parser.add_argument('-g', '--gpu', type=int, metavar='', required=False, default=0)
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
# Dataset specific
parser.add_argument('--data_dir', type=str, default=r"D:\MRI_DATASETS\Test", help='Path to dataset directory')
parser.add_argument('--test_index', type=int, default=0, help='Index of the example to use as test subject')
parser.add_argument('--z_index', type=int, default=0, help='Slice index (z_index) to extract')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import time
import numpy as np
import torch
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from inr.utils import coil_combine, path_checker, visual_mag, visual_err_mag, gen_traj, NUFFT, metrics
from inr.model import INR
from scipy import io
from CineDataset import CMRxReconToINRDataset, CineDataset

# Monkey-patch INR methods to safely handle frame_num = 1 cases
def safe_train(self, pos, kdata, e):
    timepoint = time.time()
    self.encoding.train()
    self.model.train()
    
    # Forward pass
    out = self.forward(pos, e, mask=self.mask).to(torch.float32)
    # Reshape to (1, grid_size, grid_size, frame_num, 2)
    out = out.reshape(1, self.nufft_op.grid_size, self.nufft_op.grid_size, self.nufft_op.frame_num, 2)
    # Convert to complex -> (1, grid_size, grid_size, frame_num)
    intensity = torch.view_as_complex(out)
    
    # Safe permute to (frame_num, 1, grid_size, grid_size) without using squeeze(-1)
    intensity = intensity.permute(3, 0, 1, 2)
    
    kdata_sample = self.nufft_op.forward(intensity).reshape(self.nufft_op.frame_num, self.nufft_op.coil_num, self.nufft_op.spoke_num, self.nufft_op.spoke_length)
    self.loss_train = self.cal_loss(intensity, kdata_sample, kdata)
    self.optimizer.zero_grad()
    self.loss_train.backward()
    self.optimizer.step()
    if getattr(self.scheduler, 'step_size', 0) > 0:
        self.scheduler.step()
    return (intensity, time.time() - timepoint)

def safe_infer(self, pos, img_gt, smap, sscale=1, tscale=1):
    with torch.no_grad():
        self.encoding.eval()
        self.model.eval()
        
        # Forward pass
        out = self.forward(pos, self.epoch - 1, mask=False).to(torch.float32)
        # Reshape
        grid_size = int(self.nufft_op.grid_size * sscale)
        frame_num = int(self.nufft_op.frame_num * tscale)
        out = out.reshape(1, grid_size, grid_size, frame_num, 2)
        # Convert to complex
        intensity = torch.view_as_complex(out)
        
        # Safe permute to (frame_num, 1, grid_size, grid_size) without using squeeze(-1)
        intensity = intensity.permute(3, 0, 1, 2)
        
        coil_img = intensity * smap
        combined_int = coil_combine(coil_img, smap)
        psnr, ssim = metrics(combined_int, img_gt)
    return (intensity, psnr, ssim)

# Apply monkey patches
INR.train = safe_train
INR.infer = safe_infer

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
cds = CineDataset(args.data_dir)

# Filter examples that do not contain cine_lax.mat to avoid FileNotFoundError
valid_indices = [
    i for i, ex in enumerate(cds.examples)
    if os.path.exists(os.path.join(args.data_dir, ex, "cine_lax.mat"))
]
if len(valid_indices) == 0:
    raise ValueError(f"No valid examples containing 'cine_lax.mat' found in {args.data_dir}")

if args.test_index not in valid_indices:
    fallback_index = valid_indices[0]
    print(f"Warning: requested test_index {args.test_index} is invalid (missing 'cine_lax.mat'). Falling back to index {fallback_index}.")
    args.test_index = fallback_index

ds = CMRxReconToINRDataset(
    base_dataset=cds,
    kspace_key="kspace_full",
    z_index=args.z_index,
    input_order="nxnycnznt",
    crop_square=True,
    return_torch=True,
)

# Prepare Test Subject Data
x_test = ds[args.test_index]
img_test = x_test['img'].to(device)
smap_test = x_test['smap'].to(device)
frames_test = img_test.shape[0]
coil_num_test = img_test.shape[1]
grid_size_test = img_test.shape[-1]
spoke_length_test = grid_size_test * 2
img_gt_test = coil_combine(img_test, smap_test)
scale_factor_test = torch.abs(img_gt_test).max()
img_gt_test /= scale_factor_test

ktraj_test = gen_traj(GA, spoke_length_test, frames_test * spoke_num).reshape(2, frames_test, -1).transpose(1, 0)
dcomp_test = torch.abs(torch.linspace(-1, 1, spoke_length_test)).repeat([spoke_num, 1]).to(device)
test_nufft_op = NUFFT(ktraj_test, dcomp_test, smap_test)
kdata_test = test_nufft_op.forward(img_gt_test).reshape([frames_test, coil_num_test, spoke_num, spoke_length_test])

# Initialize meta model (anchored on test dimensions)
meta_inr = INR(test_nufft_op, params, lr, relL2_eps)
meta_inr.to(device) if hasattr(meta_inr, 'to') else None

# Training pool indices (exclude test subject)
train_indices = [i for i in valid_indices if i != args.test_index]
if len(train_indices) == 0:
    # Fallback to including all if dataset only has 1 example
    train_indices = [args.test_index]

# Reptile-style meta-training loop
meta_epochs = args.meta_epochs
meta_lr = args.meta_lr
inner_steps = args.inner_steps
tasks_per_meta = args.tasks_per_meta
task_frames = args.task_frames

best_meta_psnr = 0.0
print("Starting meta-learning pretraining...")
for me in range(meta_epochs):
    # Sample tasks and accumulate adapted weights
    meta_state = {k: v.clone().detach() for k, v in meta_inr.state_dict().items()}
    for t in range(tasks_per_meta):
        # Sample a subject from training pool
        sub_idx = np.random.choice(train_indices)
        x_sub = ds[sub_idx]
        img_sub = x_sub['img'].to(device)
        smap_sub = x_sub['smap'].to(device)
        frames_sub = img_sub.shape[0]
        coil_num_sub = img_sub.shape[1]
        grid_size_sub = img_sub.shape[-1]
        spoke_length_sub = grid_size_sub * 2
        img_gt_sub = coil_combine(img_sub, smap_sub)
        scale_factor_sub = torch.abs(img_gt_sub).max()
        img_gt_sub /= scale_factor_sub

        ktraj_sub = gen_traj(GA, spoke_length_sub, frames_sub * spoke_num).reshape(2, frames_sub, -1).transpose(1, 0)
        dcomp_sub = torch.abs(torch.linspace(-1, 1, spoke_length_sub)).repeat([spoke_num, 1]).to(device)
        sub_nufft_op = NUFFT(ktraj_sub, dcomp_sub, smap_sub)
        kdata_sub = sub_nufft_op.forward(img_gt_sub).reshape([frames_sub, coil_num_sub, spoke_num, spoke_length_sub])

        # Sample task frames
        curr_task_frames = min(task_frames, frames_sub)
        idx = np.random.choice(frames_sub, curr_task_frames, replace=False)
        task_ktraj = ktraj_sub[idx]
        task_kdata = kdata_sub[idx]
        task_nufft = NUFFT(task_ktraj, dcomp_sub, smap_sub)

        # Create a fresh INR for adaptation and load meta weights
        adapt_inr = INR(task_nufft, params, lr, relL2_eps)
        adapt_inr.load_state_dict(meta_inr.state_dict())
        adapt_inr.to(device) if hasattr(adapt_inr, 'to') else None
        
        # Build positions for task
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

    # Optional evaluation on test data every few meta-iterations
    if (me + 1) % (max(1, summary_epoch // 10)) == 0 or me == meta_epochs - 1:
        with torch.no_grad():
            pos_test = meta_inr.build_pos(grid_size_test, frames_test)
            intensity, psnr_tmp, ssim_tmp = meta_inr.infer(pos_test, img_gt_test, smap_test)
        io.savemat(log_path + '/meta_proposed_{}.mat'.format(me+1), {'img_proposed': intensity.cpu().numpy()})
        visual_mag(intensity, log_path + '/meta_proposed_{}_{}_abs_{}.png'.format(spoke_num, frames_test, me+1))
        visual_err_mag(intensity, img_gt_test, log_path + '/meta_proposed_{}_{}_abs_err_{}.png'.format(spoke_num, frames_test, me+1))
        writer.add_scalar('meta_psnr', psnr_tmp, me + 1)
        writer.add_scalar('meta_ssim', ssim_tmp, me + 1)
        print('[MetaIter {}/{}] Test Subject PSNR: {:.4f} SSIM: {:.4f}'.format(me+1, meta_epochs, psnr_tmp, ssim_tmp))
        if psnr_tmp > best_meta_psnr:
            best_meta_psnr = psnr_tmp

print('Best Meta PSNR during pretraining: {:.4f}'.format(best_meta_psnr))
print('Meta-training finished.\n')

# Save model after meta-learning
meta_model_save_path = os.path.join(log_path, 'meta_model.pth')
torch.save(meta_inr.state_dict(), meta_model_save_path)
print(f'Saved meta-learned model weights to: {meta_model_save_path}\n')

# Final test subject training and inference (like in main.py)
print('Starting final adaptation and reconstruction on test subject...')
test_inr = INR(test_nufft_op, params, lr, relL2_eps)
test_inr.load_state_dict(meta_inr.state_dict())
test_inr.to(device) if hasattr(test_inr, 'to') else None
pos_test = test_inr.build_pos(grid_size_test, frames_test)

best_test_psnr = 0.0
best_test_ssim = 0.0
time_usage = 0.0
convergence_epochs = []
convergence_times = []
convergence_psnrs = []
convergence_ssims = []

epoch_loop = tqdm(range(epoch), total=epoch, leave=True)
for e in epoch_loop:
    # Training
    intensity, delta_time = test_inr.train(pos_test, kdata_test, e)
    time_usage += delta_time
    epoch_loop.set_description("[Test Train] [Lr:{:5e}]".format(test_inr.scheduler.get_last_lr()[0]))
    epoch_loop.set_postfix(dc_loss=test_inr.dc_loss.item(), tv_loss=test_inr.tv_loss.item(), max=torch.abs(intensity).max().item(),
                           lowrank_loss=test_inr.lowrank_loss.item())
    writer.add_scalar('loss_test_train', test_inr.loss_train, e + 1)

    if (e + 1) % summary_epoch == 0 or e == epoch - 1:
        with torch.no_grad():
            intensity, psnr_tmp, ssim_tmp = test_inr.infer(pos_test, img_gt_test, smap_test)
        io.savemat(log_path + '/proposed_{}.mat'.format(e+1),
                    {'img_proposed': intensity.cpu().numpy()})
        visual_mag(intensity,
            log_path + '/proposed_{}_{}_abs_{}.png'.format(spoke_num, frames_test, e+1))
        visual_err_mag(intensity, img_gt_test, log_path + '/proposed_{}_{}_abs_err_{}.png'.format(spoke_num, frames_test, e+1))
        writer.add_scalar('test_psnr', psnr_tmp, e + 1)
        writer.add_scalar('test_ssim', ssim_tmp, e + 1)
        
        convergence_epochs.append(e + 1)
        convergence_times.append(time_usage)
        convergence_psnrs.append(psnr_tmp)
        convergence_ssims.append(ssim_tmp)
        
        io.savemat(log_path + '/convergence_metrics.mat', {
            'epochs': np.array(convergence_epochs),
            'times': np.array(convergence_times),
            'psnrs': np.array(convergence_psnrs),
            'ssims': np.array(convergence_ssims)
        })
        
        if psnr_tmp > best_test_psnr:
            best_test_psnr = psnr_tmp
            best_test_ssim = ssim_tmp

print('Best Test PSNR: {:.4f}'.format(best_test_psnr))
print('Best Test SSIM: {:.4f}'.format(best_test_ssim))
print('Time Consumption: {:.2f}s'.format(time_usage))
