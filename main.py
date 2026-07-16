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
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import numpy as np
import torch
import datetime
import h5py
from tqdm import tqdm
from monai.losses import PerceptualLoss
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter
from utils import coil_combine, path_checker, visual_mag, visual_err_mag, gen_traj, NUFFT
from scipy import io
from model import INR
from CineDataset import CineDataset, CMRxReconToINRDataset
from inr.utils import metrics_extended, aggregate_dataset_metrics

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

base_path = r"/synology-data/users/naamagav/CMRxRec_for_project"
log_path = os.path.join(base_path, 'log_cmr', 'spoke{}_{}'.format(spoke_num, str(datetime.datetime.now().strftime('%y%m%d_%H%M%S'))))
path_checker(log_path)
# writer = SummaryWriter(log_path)
dataset_path = os.path.join(base_path, 'Example_dataset', 'ChallengeData_test', 'MultiCoil', 'Cine', 'TestSet', 'FullSample')
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

# Load metric models once, outside all loops
print("Loading LPIPS and Inception models...")
lpips_model = PerceptualLoss(spatial_dims=2, network_type='alex').to(device)
inception_model = models.inception_v3(weights='DEFAULT').to(device)
inception_model.fc = torch.nn.Identity()
inception_model.eval()
 
all_dataset_metrics = []

print(f"Total patients found: {len(cds)}")
 
for patient_idx in range(len(cds)):
    print(f"Loading patient {patient_idx + 1}/{len(cds)}")
    kspace, source_file = ds.load_canonical_kspace(patient_idx)  # load .mat once per patient
    nz = kspace.shape[3]
 
    patient_id = os.path.basename(os.path.dirname(source_file))
    view = "sax" if "sax" in os.path.basename(source_file) else "lax"
    print(f"Patient {patient_id} ({view}): {nz} slice(s)")
 
    for z_index in range(nz):
        tag = f"{patient_id}_{view}_slice{z_index}"
 
        is_example_slice = (patient_id == 'P002' and view == 'sax' and z_index == 2)
        print(f"  Processing slice {z_index + 1}/{nz} ({tag})")
 
        try:
            x = ds.get_slice(patient_idx, z_index, kspace=kspace)
        except Exception as e:
            print(f"  Skipping {tag}: {e}")
            continue
 
        img = x['img'][:]
        smap = x['smap'][:]

        # x = ds[1]
        # # Test is on P002 in the testset 

        # img = x['img'][:]
        # smap = x['smap'][:]

        # # Import and Preprocess Data
        # mat_path = './test_cardiac.mat'
        # with h5py.File(mat_path, 'r') as f:
        #     img = f['img'][:]
        #     smap = f['smap'][:]
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

        # Build Model and Loss
        inr = INR(nufft_op, params, lr, relL2_eps)
        pos = inr.build_pos(grid_size, frames)

        slice_writer = SummaryWriter(os.path.join(log_path, tag)) if is_example_slice else None


        psnr = 0.0
        ssim = 0.0
        time_usage = 0.0
        best_epoch = 0
        best_intensity = None
        epoch_loop = tqdm(range(epoch), total=epoch, leave=True)
        for e in epoch_loop:

            # Training
            intensity, delta_time = inr.train(pos, kdata, e)
            time_usage += delta_time
            epoch_loop.set_description("[Train] [Lr:{:5e}]".format(inr.scheduler.get_last_lr()[0]))
            epoch_loop.set_postfix(dc_loss=inr.dc_loss.item(), tv_loss=inr.tv_loss.item(), max=torch.abs(intensity).max().item(),
                                lowrank_loss=inr.lowrank_loss.item())
            # writer.add_scalar('loss_train', inr.loss_train, e + 1)
            if slice_writer is not None:
                slice_writer.add_scalar('loss_train', inr.loss_train, e + 1)

            # Infering
            if (e + 1) % summary_epoch == 0:
                with torch.no_grad():
                    intensity, psnr_tmp, ssim_tmp = inr.infer(pos, img_gt, smap)
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
                lpips_model=lpips_model, inception_model=inception_model,
            )
 
            if is_example_slice:
                io.savemat(os.path.join(log_path, f'proposed_{tag}.mat'), {'img_proposed': best_intensity.cpu().numpy()})
                visual_mag(best_intensity, os.path.join(log_path, f'proposed_{tag}_{spoke_num}_{frames}_abs_{best_epoch}.png'))
                visual_err_mag(best_intensity, img_gt, os.path.join(log_path, f'proposed_{tag}_{spoke_num}_{frames}_abs_err_{best_epoch}.png'))
 
            all_dataset_metrics.append(metrics_dict)
        else:
            print(f"  Warning: {tag} finished without running inference/evaluation.")
 
        del inr, pos, kdata, nufft_op, img, smap, img_gt
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
