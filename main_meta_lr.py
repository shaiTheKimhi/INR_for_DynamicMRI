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
parser.add_argument('--meta_epochs', type=int, default=20, help='Number of meta-training iterations')
parser.add_argument('--meta_lr', type=float, default=0.1, help='Meta learning rate (Reptile step size)')
parser.add_argument('--inner_steps', type=int, default=5, help='Inner adaptation steps per task')
parser.add_argument('--tasks_per_meta', type=int, default=4, help='Number of tasks sampled per meta-iteration')
parser.add_argument('--task_frames', type=int, default=1, help='Number of frames per task (support set size)')
# Dataset specific
parser.add_argument('--train_data_dir', type=str, default=r"D:\MRI_DATASETS\Train", help='Path to dataset directory')
parser.add_argument('--valid_data_dir', type=str, default=r"D:\MRI_DATASETS\Valid", help='Path to validation dataset directory')
parser.add_argument('--test_data_dir', type=str, default=r"D:\MRI_DATASETS\Test", help='Path to test dataset directory')
parser.add_argument('--z_index', type=int, default=2, help='Slice index (z_index) to extract')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import time
import numpy as np
import torch
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from inr.utils import coil_combine, path_checker, visual_mag, visual_err_mag, gen_traj, NUFFT, metrics, metrics_extended, \
    aggregate_dataset_metrics
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
log_path = './log_cmr_meta/spoke13_260712_135613_test'

path_checker(log_path)
writer = SummaryWriter(log_path)
meta_model_save_path = os.path.join(log_path, 'meta_model.pth')

# Import and Preprocess Data
cds_train = CineDataset(args.train_data_dir)
cds_valid = CineDataset(args.valid_data_dir)
cds_test = CineDataset(args.test_data_dir)

# Filter examples that do not contain cine_lax.mat to avoid FileNotFoundError (using test set for meta‑training pool)
valid_indices = [
    i for i, ex in enumerate(cds_valid.file_paths)
    if os.path.exists(os.path.join(cds_valid.directory, ex))
]
if len(valid_indices) == 0:
    raise ValueError(f"No valid examples containing 'cine_lax.mat' found in {args.test_data_dir}")

# Use the first valid example as the test subject
test_subject_idx = 0

train_ds = CMRxReconToINRDataset(
    base_dataset=cds_train,
    kspace_key="kspace_full",
    z_index=args.z_index,
    input_order="nxnycnznt",
    crop_square=True,
    return_torch=True,
)
test_ds = CMRxReconToINRDataset(
    base_dataset=cds_test,
    kspace_key="kspace_full",
    z_index=args.z_index,
    input_order="nxnycnznt",
    crop_square=True,
    return_torch=True,
)
import matplotlib.pyplot as plt
def show_img(img):
    plt.imshow(img.sum(axis=0).sum())
    plt.show()
# Validation dataset (separate from training and test)
valid_ds = CMRxReconToINRDataset(
    base_dataset=cds_valid,
    kspace_key="kspace_full",
    z_index=args.z_index,
    input_order="nxnycnznt",
    crop_square=True,
    return_torch=True,
)
# Reptile-style meta-training loop
meta_epochs = args.meta_epochs
meta_lr = args.meta_lr
inner_steps = args.inner_steps
tasks_per_meta = args.tasks_per_meta
task_frames = args.task_frames
# Use first example for validation
val_subject_idx = 0
x_val = valid_ds[val_subject_idx]
img_val = x_val['img'].to(device)
smap_val = x_val['smap'].to(device)
frames_val = img_val.shape[0]
coil_num_val = img_val.shape[1]
grid_size_val = img_val.shape[-1]
spoke_length_val = grid_size_val * 2
img_gt_val = coil_combine(img_val, smap_val)
scale_factor_val = torch.abs(img_gt_val).max()
img_gt_val /= scale_factor_val
ktraj_val = gen_traj(GA, spoke_length_val, frames_val * spoke_num).reshape(2, frames_val, -1).transpose(1, 0)
dcomp_val = torch.abs(torch.linspace(-1, 1, spoke_length_val)).repeat([spoke_num, 1]).to(device)
val_nufft_op = NUFFT(ktraj_val, dcomp_val, smap_val)
kdata_val = val_nufft_op.forward(img_gt_val).reshape([frames_val, coil_num_val, spoke_num, spoke_length_val])
# Initialize meta model (anchored on test dimensions)
meta_inr = INR(val_nufft_op, params, meta_lr, relL2_eps)
meta_inr.to(device) if hasattr(meta_inr, 'to') else None

# Training pool indices (exclude test subject)
train_indices = [i for i in valid_indices if i != test_subject_idx]
if len(train_indices) == 0:
    train_indices = [test_subject_idx]


# Remove early‑stopping variables
best_meta_psnr = 0.0
# patience_counter and meta_patience are no longer used

print("Starting meta-learning pretraining...")
for me in range(meta_epochs):
    # Sample tasks and accumulate adapted weights
    meta_state = {k: v.clone().detach() for k, v in meta_inr.state_dict().items()}
    for t in range(tasks_per_meta):
        # Sample a subject from training pool
        sub_idx = np.random.choice(train_indices)
        try:
            x_sub = train_ds[sub_idx]
        except:
            continue

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

    # Optional evaluation on validation data every few meta-iterations
    if (me + 1) % (max(1, summary_epoch // 10)) == 0 or me == meta_epochs - 1:
        # Validation on separate validation dataset
        pos_val = meta_inr.build_pos(grid_size_val, frames_val)
        intensity_val, psnr_val, ssim_val = meta_inr.infer(pos_val, img_gt_val, smap_val)
        io.savemat(log_path + '/meta_val_{}.mat'.format(me+1), {'img_proposed': intensity_val.cpu().numpy()})
        visual_mag(intensity_val, log_path + '/meta_val_{}_{}_abs_{}.png'.format(spoke_num, frames_val, me+1))
        visual_err_mag(intensity_val, img_gt_val, log_path + '/meta_val_{}_{}_abs_err_{}.png'.format(spoke_num, frames_val, me+1))
        writer.add_scalar('meta_val_psnr', psnr_val, me + 1)
        writer.add_scalar('meta_val_ssim', ssim_val, me + 1)
        print('[MetaIter {}/{}] Validation PSNR: {:.4f} SSIM: {:.4f}'.format(me+1, meta_epochs, psnr_val, ssim_val))
        if psnr_val > best_meta_psnr:
            print(f"Best epoch: {epoch}, PSNR: {psnr_val}, SSIM: {ssim_val}")
            best_meta_psnr = psnr_val
            torch.save(meta_inr.state_dict(), meta_model_save_path)
print('Meta-training finished.\n')

torch.cuda.empty_cache()

# Final test subject training and inference (like in main.py)
print('Starting final adaptation and reconstruction on test subject...')


all_dataset_metrics = []

for patient_idx in range(len(cds_test)):
    print(f"Loading patient {patient_idx + 1}/{len(cds_test)}")
    kspace, source_file = test_ds.load_canonical_kspace(patient_idx)  # load .mat once per patient
    nz = kspace.shape[3]

    ## the training was on sax, z_index=2
    patient_id = os.path.basename(os.path.dirname(source_file))
    view = "sax" if "sax" in os.path.basename(source_file) else "lax"
    if view == "lax":
        continue
    print(f"Patient {patient_id} ({view}): {nz} slice(s)")

    for z_index in range(nz):
        tag = f"{patient_id}_{view}_slice{z_index}"

        is_example_slice = (patient_id == 'P002' and view == 'sax' and z_index == 2)
        print(f"  Processing slice {z_index + 1}/{nz} ({tag})")

        try:
            x = test_ds.get_slice(patient_idx, z_index, kspace=kspace)
        except Exception as e:
            print(f"  Skipping {tag}: {e}")
            continue

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
        ktraj = gen_traj(GA, spoke_length, frames * spoke_num).reshape(2, frames, -1).transpose(1, 0)
        dcomp = torch.abs(torch.linspace(-1, 1, spoke_length)).repeat([spoke_num, 1]).to(device)
        nufft_op = NUFFT(ktraj, dcomp, smap)
        kdata = nufft_op.forward(img_gt).reshape([frames, coil_num, spoke_num, spoke_length])

        # Build Test Model and Loss
        test_inr = INR(nufft_op, params, lr, relL2_eps)
        test_inr.load_state_dict(torch.load("log_cmr_meta/spoke13_260712_135613/meta_model.pth"))
        pos = test_inr.build_pos(grid_size, frames)

        slice_writer = SummaryWriter(os.path.join(log_path, tag)) if is_example_slice else None


        psnr = 0.0
        ssim = 0.0
        time_usage = 0.0
        best_epoch = 0
        best_intensity = None
        epoch_loop = tqdm(range(epoch), total=epoch, leave=True)
        for e in epoch_loop:

            # Training
            intensity, delta_time = test_inr.train(pos, kdata, e)
            time_usage += delta_time
            epoch_loop.set_description("[Train] [Lr:{:5e}]".format(test_inr.scheduler.get_last_lr()[0]))
            epoch_loop.set_postfix(dc_loss=test_inr.dc_loss.item(), tv_loss=test_inr.tv_loss.item(), max=torch.abs(intensity).max().item(),
                                   lowrank_loss=test_inr.lowrank_loss.item())
            # writer.add_scalar('loss_train', inr.loss_train, e + 1)
            if slice_writer is not None:
                slice_writer.add_scalar('loss_train', test_inr.loss_train, e + 1)

            # Infering
            if (e + 1) % summary_epoch == 0:
                with torch.no_grad():
                    intensity, psnr_tmp, ssim_tmp = test_inr.infer(pos, img_gt, smap)
                if is_example_slice:
                    io.savemat(log_path + '/proposed_{}_{}.mat'.format(e+1, tag),
                                {'img_proposed': intensity.cpu().numpy()})
                    visual_mag(intensity,
                        log_path + '/proposed_{}_{}_abs_{}_{}.png'.format(spoke_num, frames, e+1, tag))
                    visual_err_mag(intensity, img_gt, log_path + '/proposed_{}_{}_abs_err_{}_{}.png'.format(spoke_num, frames, e+1, tag))

                if slice_writer is not None:
                    slice_writer.add_scalar('psnr', psnr_tmp, e + 1)
                    slice_writer.add_scalar('ssim', ssim_tmp, e + 1)

                if psnr_tmp > psnr:
                    psnr = psnr_tmp
                    ssim = ssim_tmp
                    best_intensity = intensity.clone().detach()
                    best_epoch = e + 1

        if slice_writer is not None:
            slice_writer.close()

        # Summary
        print(f'--- {tag} Training Complete ---')
        print('Best PSNR: {:.4f}'.format(psnr))
        print('SSIM: {:.4f}'.format(ssim))
        print('Best Epoch: {}'.format(best_epoch))
        print('Time Consumption: {:.2f}s'.format(time_usage))


        if best_intensity is not None:
            metrics_filename = os.path.join(log_path, f'proposed_{tag}_{spoke_num}_{frames}_abs_err_metrics.txt')
            metrics_dict = metrics_extended(
                best_intensity, img_gt, time_usage, metrics_filename if is_example_slice else None,
            )

            if is_example_slice:
                io.savemat(os.path.join(log_path, f'proposed_{tag}.mat'), {'img_proposed': best_intensity.cpu().numpy()})
                visual_mag(best_intensity, os.path.join(log_path, f'proposed_{tag}_{spoke_num}_{frames}_abs_{best_epoch}.png'))
                visual_err_mag(best_intensity, img_gt, os.path.join(log_path, f'proposed_{tag}_{spoke_num}_{frames}_abs_err_{best_epoch}.png'))

            all_dataset_metrics.append(metrics_dict)
        else:
            print(f"  Warning: {tag} finished without running inference/evaluation.")

        del test_inr, pos, kdata, nufft_op, img, smap, img_gt
        if best_intensity is not None:
            del best_intensity
        torch.cuda.empty_cache()

    del kspace  # free the full patient volume before moving to next patient

# Final Aggregated Metrics Summary
print("\nCalculating final aggregated statistics...")
if all_dataset_metrics:
    final_log_path = os.path.join(log_path, 'Final_INR_metrics.txt')
    aggregate_dataset_metrics(all_dataset_metrics, final_log_path)
else:
    print("No slices were successfully processed.")

print("Done!")
