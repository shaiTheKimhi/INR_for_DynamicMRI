
import torch
from torch import fft
import torchkbnufft as tkbn
import numpy as np
import os
from typing import List, Tuple, Union, Optional
import h5py
from matplotlib import cm
from torchvision.utils import make_grid
import imageio as imgio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from monai.metrics import FIDMetric, SSIMMetric, PSNRMetric, compute_frechet_distance
from monai.losses import PerceptualLoss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
jet_cmap = cm.get_cmap('jet')

def fftnc(x: torch.Tensor, dim: Optional[List[int]]=None) -> torch.Tensor:
    """
    N-dim centered FFT

    :param x: input N-dim Tensor (CPU/GPU)
    :param dim: run FFT in given dim
    :return: output N-dim Tensor (CPU/GPU)
    """
    device = x.device
    if dim is None:
        dim = [0] * x.dim()
        for i in range(1, x.dim()):
            dim[i] = i
    return fft.ifftshift(fft.fftn(fft.fftshift(x, dim=dim), dim=dim), dim=dim)

def ifftnc(x: torch.Tensor, dim: Optional[List[int]]=None) -> torch.Tensor:
    """
    N-dim centered iFFT

    :param x: input N-dim Tensor (CPU/GPU)
    :param dim: run iFFT in given dim
    :return: output N-dim Tensor (CPU/GPU)
    """
    device = x.device
    if dim is None:
        dim = [0] * x.dim()
        for i in range(1, x.dim()):
            dim[i] = i
    return fft.fftshift(fft.ifftn(fft.ifftshift(x, dim=dim), dim=dim), dim=dim)

def coil_combine(coil_img, smap):
    img_combined = torch.sum(smap.conj() * coil_img, dim=1, keepdim=True)
    img_combined[torch.isinf(img_combined)] = 0
    img_combined[torch.isnan(img_combined)] = 0
    return img_combined

def normalization(img, max=None, min=None):
    if max is None:
        max = img.max()
    else:
        max = img.max() * max
    if min is None:
        min = img.min()
    else:
        min = img.min() * min
    if torch.is_tensor(img):
        img = img.clamp(min=min, max=max)
        img = (img - min) / (max - min)
        img = img.cpu()
        return img
    img = np.clip(img, min, max)
    img = (img - min) / (max - min)
    return img

def path_checker(path: str):
    """
    Check if the path exists
    """
    if not os.path.isdir(path):
        os.makedirs(path)
        print(path + ' not exists, already created')
    else:
        print(path + ' exists')

def visual_mag(img: torch.Tensor, path: str, max_value: Optional[float]=0.6, nrow_num=6):
    assert img.shape[1] == 1
    img_grid = make_grid(normalization(torch.abs(img), max_value), nrow=nrow_num, pad_value=1.0).cpu().numpy().transpose(1, 2, 0)
    imgio.imsave(path, np.uint8(img_grid * 255))

def visual_phi(img: torch.Tensor, path: str, nrow_num=6):
    assert img.shape[1] == 1
    img_grid = make_grid(torch.angle(img), nrow=nrow_num, normalize=True, value_range=(-torch.pi, torch.pi), pad_value=1.0).cpu().numpy().transpose(1, 2, 0)
    imgio.imsave(path, np.uint8(img_grid * 255))

def visual_gif(img: torch.Tensor, path: str, max_value: Optional[float]=0.6):
    assert img.shape[1] == 1
    imgio.mimsave(path, [normalization(torch.abs(img), max_value)[i, ...].squeeze() for i in range(img.shape[0])], duration=0.1)

def visual_err_mag(img1: torch.Tensor, img2: torch.Tensor, path: str, nrow_num=6, max_value=0.1):
    assert img1.shape[1] == 1 and img2.shape[1] == 1
    img_grid = make_grid(normalization(torch.abs(img1)), nrow=nrow_num, pad_value=2.0) - make_grid(normalization(torch.abs(img2)), nrow=nrow_num)
    mask = np.stack([(img_grid[0, :, :] == 2).cpu().numpy()] * 4, -1)
    jet_img = jet_cmap(np.clip(torch.abs(img_grid[0, :, :]).cpu().numpy(), 0, max_value) / max_value)
    jet_img[mask] = 1
    imgio.imsave(path, np.uint8(jet_img * 255))

def visual_err_phi(img1: torch.Tensor, img2: torch.Tensor, path: str, nrow_num=6, max_value=torch.pi / 6):
    assert img1.shape[1] == 1 and img2.shape[1] == 1
    img_grid = make_grid(torch.angle(img1), normalize=True, value_range=(-torch.pi, torch.pi), nrow=nrow_num, pad_value=2.0) - make_grid(torch.angle(img2), normalize=True, value_range=(-torch.pi, torch.pi), nrow=nrow_num)
    mask = np.stack([(img_grid[0, :, :] == 2).cpu().numpy()] * 4, -1)
    jet_img = jet_cmap(np.clip(torch.abs(img_grid[0, :, :]).cpu().numpy(), 0, max_value / (2 * torch.pi)) / max_value * (2 * torch.pi))
    jet_img[mask] = 1
    imgio.imsave(path, np.uint8(jet_img * 255))

def gen_traj(theta, spoke_length, spoke_num, ind=0):
    angles = theta * torch.arange(ind, ind + spoke_num, dtype=torch.float32, device=device).unsqueeze_(1)
    pos = torch.linspace(-torch.pi, torch.pi, spoke_length, device=device).unsqueeze_(0)
    kx = torch.mm(torch.cos(angles), pos)
    ky = torch.mm(torch.sin(angles), pos)
    return torch.stack((kx.flatten(), ky.flatten()))

class TVLoss(torch.nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        return torch.sum(torch.abs(x[1:, :, :, :] - x[:x.shape[0] - 1, :, :, :])) / x.numel()

class STVLoss(torch.nn.Module):

    def __init__(self):
        super(STVLoss, self).__init__()

    def forward(self, x):
        # return torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :x.shape[-2] - 1, :])) / x.numel() + torch.sum(torch.abs(x[:, :, :, :x.shape[-1] - 1])) / x.numel()

        return torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :x.shape[-2] - 1, :])) / x.numel() + \
            torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :x.shape[-1] - 1])) / x.numel()

class RelL2Loss(torch.nn.Module):

    def __init__(self, rel=True, eps=0.0001):
        super(RelL2Loss, self).__init__()
        self.eps = eps
        self.rel = rel

    def forward(self, input, label):
        if self.rel:
            loss = (label.real - input.real) ** 2 / (input.real.detach() ** 2 + self.eps) + (label.imag - input.imag) ** 2 / (input.imag.detach() ** 2 + self.eps)
            return loss
        loss = ((label.real - input.real) ** 2 + (label.imag - input.imag) ** 2) * 100
        return loss

class LRLoss(torch.nn.Module):

    def __init__(self):
        super(LRLoss, self).__init__()

    def forward(self, x):
        return torch.abs(torch.linalg.norm(x.reshape(x.shape[0], -1), 'nuc')) / x.shape[0]

class NUFFT:

    def __init__(self, ktraj, dcomp, smap):
        self.ktraj = ktraj
        self.dcomp = dcomp
        self.smap = smap
        self.frame_num = self.ktraj.shape[0]
        self.spoke_num, self.spoke_length = self.dcomp.shape
        self.coil_num = self.smap.shape[0]
        self.grid_size = self.spoke_length // 2
        self.nufft_op = tkbn.KbNufft(im_size=(self.grid_size, self.grid_size)).to(torch.complex64).to(device)
        self.nufft_adj_op = tkbn.KbNufftAdjoint(im_size=(self.grid_size, self.grid_size)).to(torch.complex64).to(device)

    def forward(self, img):
        if img.shape[1] == 1:
            return self.nufft_op(img, self.ktraj, smaps=self.smap).reshape([self.frame_num, self.coil_num, self.spoke_num, self.spoke_length]) / self.grid_size
        return self.nufft_op(img, self.ktraj).reshape([self.frame_num, self.coil_num, self.spoke_num, self.spoke_length]) / self.grid_size

    def adjoint(self, kdata):
        return self.nufft_adj_op(kdata.reshape(self.frame_num, self.coil_num, -1) * self.dcomp.flatten(), self.ktraj, smaps=self.smap) / (self.grid_size * torch.pi / (2 * self.spoke_num)) / (torch.abs(self.smap) ** 2).sum(dim=0).unsqueeze(0).unsqueeze(0)

def metrics(imgs: torch.Tensor, gts: torch.Tensor, file_path: Optional[str]=None):
    frames = imgs.shape[0]
    imgs = normalization(torch.abs(imgs.squeeze())).cpu().numpy()
    gts = normalization(torch.abs(gts.squeeze())).cpu().numpy()
    psnr = [peak_signal_noise_ratio(imgs[i, ...].squeeze(), gts[i, ...].squeeze()) for i in range(frames)]
    ssim = [structural_similarity(imgs[i, ...].squeeze(), gts[i, ...].squeeze(), data_range=1.0) for i in range(frames)]
    if file_path is not None:
        with open(file_path, 'w') as f:
            for i in range(frames):
                f.writelines('Frame {}\t{:6f}\t{:6f}\n'.format(i + 1, psnr[i], ssim[i]))
            f.writelines('Mean\t{:6f}\t{:6f}\n'.format(np.mean(psnr), np.mean(ssim)))
            f.writelines('Std\t{:6f}\t{:6f}\n'.format(np.std(psnr), np.std(ssim)))
    return (np.mean(psnr), np.mean(ssim))