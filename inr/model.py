import torch
import time
from torch.optim import lr_scheduler
import tinycudann as tcnn
from tqdm import tqdm
from utils import coil_combine, TVLoss, RelL2Loss, STVLoss, LRLoss, metrics
class INR(torch.nn.Module):
    def __init__(self, nufft_op, params, lr, eps):
        super(INR, self).__init__()
        self.encoding_config = {'otype': 'HashGrid', 'n_levels': params['n_levels'], 'n_features_per_level': params['n_features_per_level'], 'log2_hashmap_size': params['log2_hashmap_size'], 'base_resolution': params['base_resolution'], 'per_level_scale': params['per_level_scale']}
        self.network_config = {'otype': 'FullyFusedMLP', 'activation': 'ReLU', 'output_activation': 'None', 'n_neurons': params['n_neurons'], 'n_hidden_layers': params['n_hidden_layers']}
        self.nufft_op = nufft_op
        self.epoch = params['epochs']
        self.epochs_per_level = self.epoch // params['n_levels']
        self.relL2 = params['relL2']
        self.mask = params['mask']
        self.tv_weight = params['tv_weight']
        self.stv_weight = params['stv_weight']
        self.lr_weight = params['lr_weight']
        self.encoding = tcnn.Encoding(n_input_dims=3, encoding_config=self.encoding_config)
        self.model = tcnn.Network(n_input_dims=self.encoding.n_output_dims, n_output_dims=2, network_config=self.network_config)
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': lr, 'weight_decay': 1e-06}, {'params': self.encoding.parameters(), 'lr': lr, 'weight_decay': 0}])
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.epoch // 2, gamma=0.5)
        self.DC_loss = RelL2Loss(self.relL2, eps=eps)
        self.TV_loss = TVLoss()
        self.STV_loss = STVLoss()
        self.LR_loss = LRLoss()
    def build_pos(self, grid_size, frame_num):
        xs = torch.linspace(1 / (2 * grid_size), 1 - 1 / (2 * grid_size), grid_size, device='cuda')
        ys = torch.linspace(1 / (2 * grid_size), 1 - 1 / (2 * grid_size), grid_size, device='cuda')
        ts = torch.linspace(1 / (2 * grid_size), 1 - 1 / (2 * grid_size), grid_size, device='cuda')[:frame_num]
        xv, yv, tv = torch.meshgrid([xs, ys, ts], indexing='ij')
        pos = torch.stack((tv.flatten(), yv.flatten(), xv.flatten())).t()
        return pos
    def build_ssr_pos(self, grid_size, frame_num, scale):
        xs = torch.linspace(1 / (2 * grid_size), 1 - 1 / (2 * grid_size), grid_size, device='cuda')
        ys = torch.linspace(1 / (2 * grid_size), 1 - 1 / (2 * grid_size), grid_size, device='cuda')
        ts = torch.linspace(1 / (2 * grid_size), 1 - 1 / (2 * grid_size), int(scale * grid_size), device='cuda')[:frame_num]
        xv, yv, tv = torch.meshgrid([xs, ys, ts], indexing='ij')
        pos = torch.stack((tv.flatten(), yv.flatten(), xv.flatten())).t()
        xs_dense = torch.linspace(1 / (2 * grid_size), 1 - 1 / (2 * grid_size), int(scale * grid_size), device='cuda')
        ys_dense = torch.linspace(1 / (2 * grid_size), 1 - 1 / (2 * grid_size), int(scale * grid_size), device='cuda')
        xv_dense, yv_dense, tv_dense = torch.meshgrid([xs_dense, ys_dense, ts], indexing='ij')
        pos_dense_s = torch.stack((tv_dense.flatten(), yv_dense.flatten(), xv_dense.flatten())).t()
        return (pos, pos_dense_s)
    def build_tsr_pos(self, grid_size, frame_num, scale):
        xs = torch.linspace(1 / (2 * grid_size), 1 - 1 / (2 * grid_size), grid_size, device='cuda')
        ys = torch.linspace(1 / (2 * grid_size), 1 - 1 / (2 * grid_size), grid_size, device='cuda')
        ts_dense = torch.linspace(1 / (2 * grid_size), 1 - 1 / (2 * grid_size), int(scale * grid_size), device='cuda')[:frame_num * scale]
        ts = ts_dense[::scale]
        xv, yv, tv = torch.meshgrid([xs, ys, ts], indexing='ij')
        pos = torch.stack((tv.flatten(), yv.flatten(), xv.flatten())).t()
        xv_dense, yv_dense, tv_dense = torch.meshgrid([xs, ys, ts_dense], indexing='ij')
        pos_dense_t = torch.stack((tv_dense.flatten(), yv_dense.flatten(), xv_dense.flatten())).t()
        return (pos, pos_dense_t)
    def forward(self, input, e, mask):
        enc = self.encoding(input.reshape((-1), 3))
        if mask:
            enc_mask = torch.zeros_like(enc).to('cuda')
            enc_mask[:, :(e // self.epochs_per_level + 1) * self.encoding_config['n_features_per_level']] = torch.tensor(1.0)
            enc = enc * enc_mask
        return self.model(enc)
    def cal_loss(self, intensity, kdata_sample, kdata):
        self.tv_loss = (self.TV_loss(intensity.real) + self.TV_loss(intensity.imag)) / torch.abs(intensity.detach()).max()
        self.stv_loss = (self.STV_loss(intensity.real) + self.STV_loss(intensity.imag)) / torch.abs(intensity.detach()).max()
        self.lowrank_loss = self.LR_loss(intensity)
        self.dc_loss = self.DC_loss(kdata_sample, kdata).mean()
        return self.dc_loss + self.tv_weight * self.tv_loss + self.lr_weight * self.lowrank_loss + self.stv_weight * self.stv_loss
    def train(self, pos, kdata, e):
        timepoint = time.time()
        self.encoding.train()
        self.model.train()
        intensity = torch.view_as_complex(self.forward(pos, e, mask=self.mask).to(torch.float32).reshape(1, self.nufft_op.grid_size, self.nufft_op.grid_size, self.nufft_op.frame_num, 2)).squeeze((-1)).permute(3, 0, 1, 2)
        kdata_sample = self.nufft_op.forward(intensity).reshape(self.nufft_op.frame_num, self.nufft_op.coil_num, self.nufft_op.spoke_num, self.nufft_op.spoke_length)
        self.loss_train = self.cal_loss(intensity, kdata_sample, kdata)
        self.optimizer.zero_grad()
        self.loss_train.backward()
        self.optimizer.step()
        self.scheduler.step()
        return (intensity, time.time() - timepoint)
    def infer(self, pos, img_gt, smap, sscale=1, tscale=1):
        with torch.no_grad():
            self.encoding.eval()
            self.model.eval()
            intensity = torch.view_as_complex(self.forward(pos, self.epoch - 1, mask=False).to(torch.float32).reshape(1, int(self.nufft_op.grid_size * sscale), int(self.nufft_op.grid_size * sscale), int(self.nufft_op.frame_num * tscale), 2)).squeeze((-1)).permute(3, 0, 1, 2)
            coil_img = intensity * smap
            combined_int = coil_combine(coil_img, smap)
            psnr, ssim = metrics(combined_int, img_gt)
        return (intensity, psnr, ssim)