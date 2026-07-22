import os
import argparse
import time
import datetime
import numpy as np
import torch
from scipy import io
from monai.losses import PerceptualLoss
import torchvision.models as models

# Parse only the essential arguments for the naive reconstruction
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--spokes', type=int, metavar='', required=True)
parser.add_argument('-g', '--gpu', type=int, metavar='', required=True)
parser.add_argument('-d', '--data_dir', type=str, metavar='', required=False, default=r"D:\MRI_DATASETS\Test")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

from utils import coil_combine, path_checker, visual_mag, visual_err_mag, gen_traj, NUFFT
from CineDataset import CineDataset, CMRxReconToINRDataset
from inr.utils import metrics_extended, aggregate_dataset_metrics

# Important Constants
GA = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))  # GoldenAngle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
spoke_num = args.spokes

# Setup Paths
log_path = './log_cmr/naive_nufft_spoke{}_{}'.format(spoke_num, str(datetime.datetime.now().strftime('%y%m%d_%H%M%S')))
path_checker(log_path)

dataset_path = args.data_dir
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
all_dataset_metrics = []

print(f"Total samples found: {len(ds)}")
# x = ds[1]
# Test is on P002 in the testset 

# Load metric models once, outside the loop
print("Loading LPIPS and Inception models...")
lpips_model = PerceptualLoss(spatial_dims=2, network_type='alex').to(device)
inception_model = models.inception_v3(weights='DEFAULT').to(device)
inception_model.fc = torch.nn.Identity()
inception_model.eval()

for patient_idx in range(len(cds)):
    print(f"Loading patient {patient_idx + 1}/{len(cds)}")
    kspace, source_file = ds.load_canonical_kspace(patient_idx)   # load .mat once per patient
    nz = kspace.shape[3]

    # e.g. ".../P002/cine_sax.mat" -> patient_id="P002", view="sax"
    patient_id = os.path.basename(os.path.dirname(source_file))
    view = "sax" if "sax" in os.path.basename(source_file) else "lax"

    print(f"Patient {patient_id} ({view}): {nz} slice(s)")

    for z_index in range(nz):

        print(f"  Processing slice {z_index + 1}/{nz}")
        try:
            x = ds.get_slice(patient_idx, z_index, kspace=kspace)
        except Exception as e:
            print(f"  Skipping patient {patient_idx} slice {z_index}: {e}")
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
        img_gt /= scale_factor

        ktraj = gen_traj(GA, spoke_length, frames * spoke_num).reshape(2, frames, -1).transpose(1, 0)
        dcomp = torch.abs(torch.linspace(-1, 1, spoke_length)).repeat([spoke_num, 1]).to(device)
        nufft_op = NUFFT(ktraj, dcomp, smap)

        raw_kdata = nufft_op.forward(img_gt)

        start_time = time.time()
        recon = nufft_op.adjoint(raw_kdata)
        time_usage = time.time() - start_time

        tag = f"{patient_id}_{view}_slice{z_index}"
        metrics_filename = os.path.join(log_path, f'naive_{tag}_{spoke_num}_{frames}_abs_err_metrics.txt')

        if patient_id == 'P002' and view == 'sax' and z_index == 2:

            io.savemat(os.path.join(log_path, f'naive_recon_{tag}.mat'), {'img_naive': recon.cpu().numpy()})

            metrics_dict = metrics_extended(
                recon, img_gt, time_usage, metrics_filename,
                lpips_model=lpips_model, inception_model=inception_model,
            )

            visual_mag(recon, os.path.join(log_path, f'naive_{tag}_{spoke_num}_{frames}_abs.png'))
            visual_err_mag(recon, img_gt, os.path.join(log_path, f'naive_{tag}_{spoke_num}_{frames}_abs_err.png'))
        else:
            metrics_dict = metrics_extended(
                recon, img_gt, time_usage,
                lpips_model=lpips_model, inception_model=inception_model,
            )   

        all_dataset_metrics.append(metrics_dict)

        del raw_kdata, recon, img_gt, img, smap, nufft_op
        torch.cuda.empty_cache()

    del kspace  # free the full patient volume before moving to next patient

# Final Aggregated Metrics Summary
print("\nCalculating final aggregated statistics...")
if all_dataset_metrics:
    final_log_path = os.path.join(log_path, 'Final_nufft_metrics.txt')
    aggregate_dataset_metrics(all_dataset_metrics, final_log_path)
else:
    print("No files were successfully processed.")

print("Done!")