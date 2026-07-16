import os
import argparse
import time
import datetime
import numpy as np
import torch
import h5py
from tqdm import tqdm
from monai.losses import PerceptualLoss
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter
from utils import coil_combine, path_checker, visual_mag, visual_err_mag, gen_traj, NUFFT
from scipy import io
from CineDataset import CineDataset, CMRxReconToINRDataset
from inr.utils import metrics_extended, aggregate_dataset_metrics
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

        # x = ds[1]
        # # Test is on P002 in the testset 

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
        slice_writer = SummaryWriter(os.path.join(log_path, tag)) if is_example_slice else None


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
            
            if slice_writer is not None:
                slice_writer.add_scalar('loss_total', loss.item(), i + 1)
                slice_writer.add_scalar('loss_dc', dc_loss_val.item(), i + 1)
                slice_writer.add_scalar('loss_tv', tv_loss_val.item(), i + 1)

            # Inferring / Evaluation
            if (i + 1) % summary_epoch == 0:
                with torch.no_grad():

                    intensity_eval = recon_img.clone().detach()   

                    coil_img = intensity_eval * smap 
                    combined_int = coil_combine(coil_img, smap)
                    
                    # Calculate metrics
                    psnr_tmp, ssim_tmp = metrics(combined_int, img_gt)
                
                    if is_example_slice:
                        io.savemat(log_path + '/proposed_{}_{}.mat'.format(i+1, tag),
                                    {'img_proposed': intensity_eval.cpu().numpy()})
                        visual_mag(intensity_eval,
                            log_path + '/proposed_{}_{}_abs_{}_{}.png'.format(spoke_num, frames, i+1, tag))
                        visual_err_mag(intensity_eval, img_gt, log_path + '/proposed_{}_{}_abs_err_{}_{}.png'.format(spoke_num, frames, i+1, tag))

                    if slice_writer is not None:
                        slice_writer.add_scalar('psnr', psnr_tmp, i + 1)
                        slice_writer.add_scalar('ssim', ssim_tmp, i + 1)
                    
                    if psnr_tmp > psnr:
                        psnr = psnr_tmp
                        ssim = ssim_tmp
                        best_intensity = intensity_eval.clone().detach()
                        best_epoch = i + 1
        if slice_writer is not None:
            slice_writer.close()

        # Summary
        print(f'--- {tag} Optimization Complete ---')
        print('Best PSNR: {:.4f}'.format(psnr))
        print('SSIM: {:.4f}'.format(ssim))
        print('Best Iteration: {}'.format(best_epoch))
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

        del recon_img, optimizer, kdata, nufft_op, img, smap, img_gt
        if best_intensity is not None:
            del best_intensity
        torch.cuda.empty_cache()

    del kspace  # free the full patient volume before moving to next patient

# Final Aggregated Metrics Summary
print("\nCalculating final aggregated statistics...")
if all_dataset_metrics:
    final_log_path = os.path.join(log_path, 'Final_GRASP_metrics.txt')
    aggregate_dataset_metrics(all_dataset_metrics, final_log_path)
else:
    print("No slices were successfully processed.")

print("Done!")