import os
import numpy as np
from mat73 import loadmat
import torch
from torch.utils.data import Dataset, DataLoader


class CineDataset(Dataset):
    def __init__(self, directory: str):
        super().__init__()
        self.directory = directory
        self.examples = os.listdir(directory)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]
        ## load .mat file
        arr = loadmat(os.path.join(self.directory, example, "cine_lax.mat"))
        return arr



def to_complex_numpy(x):
    """
    Robust conversion for possible MATLAB/HDF5 complex formats.

    Supports:
    1. Already-complex numpy arrays.
    2. Compound dtype with fields ['real', 'imag'].
    3. Real/imag stacked in the last dimension [..., 2].
    """
    x = np.asarray(x)

    if np.iscomplexobj(x):
        return x

    if x.dtype.fields is not None:
        fields = x.dtype.fields.keys()
        if "real" in fields and "imag" in fields:
            return x["real"] + 1j * x["imag"]

    if x.shape[-1] == 2:
        return x[..., 0] + 1j * x[..., 1]

    raise ValueError(
        f"Cannot infer complex format from array with shape {x.shape} "
        f"and dtype {x.dtype}."
    )


def ifft2c(kspace, axes=(0, 1)):
    """
    Centered 2D inverse FFT.

    Expects spatial k-space axes to be axes=(0, 1), e.g.
        kspace: [nx, ny, nc, nt]
    """
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace, axes=axes),
            axes=axes,
            norm="ortho",
        ),
        axes=axes,
    )


def center_crop_square(x):
    """
    Center-crop the last two dimensions to square.

    Input:
        x: [..., H, W]

    Output:
        x: [..., N, N]
    """
    h, w = x.shape[-2], x.shape[-1]
    n = min(h, w)

    h0 = (h - n) // 2
    w0 = (w - n) // 2

    return x[..., h0:h0 + n, w0:w0 + n]


def estimate_smap_from_rss(img_tchw, eps=1e-8):
    """
    Estimate simple coil sensitivity maps from coil images.

    Input:
        img_tchw: [T, C, H, W] complex

    Output:
        smap: [C, H, W] complex

    This is a simple RSS-normalized estimate:
        smap_c = coil_ref_c / RSS(coil_ref)

    For publication-quality reconstruction, replace this with ESPIRiT/BART/sigpy maps.
    """
    ref = img_tchw.mean(axis=0)  # [C, H, W]
    rss = np.sqrt(np.sum(np.abs(ref) ** 2, axis=0, keepdims=True)) + eps
    smap = ref / rss
    return smap.astype(np.complex64)


class CMRxReconToINRDataset(Dataset):
    """
    Wrapper dataset that converts CMRxRecon kspace_full into INR_for_DynamicMRI-style
    img and smap fields.

    Assumed canonical CMRxRecon k-space shape:
        kspace_full: [nx, ny, nc, nz, nt]

    Returned:
        sample["img"]  : [T, C, H, W] complex64
        sample["smap"] : [C, H, W] complex64
    """

    def __init__(
        self,
        base_dataset,
        kspace_key="kspace_full",
        z_index=0,
        input_order="nxnycnznt",
        crop_square=True,
        return_torch=True,
        keep_original_item=False,
    ):
        """
        Args:
            base_dataset:
                Existing dataset whose __getitem__ returns either:
                    kspace_full
                or:
                    {"kspace_full": kspace_full, ...}

            kspace_key:
                Key to read from base item if the base item is a dict.

            z_index:
                Which SAX slice / LAX view to extract from nz.

            input_order:
                "nxnycnznt" means [nx, ny, nc, nz, nt].
                "ntnzncnynx" can be used if your helper gives HDF5-reversed arrays.

            crop_square:
                If True, crop spatial dimensions to H == W.
                The original INR code assumes a square grid.

            return_torch:
                If True, return torch complex tensors.
                If False, return numpy complex arrays.

            keep_original_item:
                If True, keep non-kspace metadata from the base dataset item.
        """
        self.base_dataset = base_dataset
        self.kspace_key = kspace_key
        self.z_index = z_index
        self.input_order = input_order
        self.crop_square = crop_square
        self.return_torch = return_torch
        self.keep_original_item = keep_original_item

    def __len__(self):
        return len(self.base_dataset)

    def _extract_kspace(self, item):
        if isinstance(item, dict):
            if self.kspace_key not in item:
                raise KeyError(
                    f"Expected key '{self.kspace_key}' in base dataset item. "
                    f"Available keys: {list(item.keys())}"
                )
            return item[self.kspace_key]
        return item

    def _canonicalize_order(self, kspace):
        """
        Return kspace as [nx, ny, nc, nz, nt].
        """
        if self.input_order == "nxnycnznt":
            return kspace

        if self.input_order == "ntnzncnynx":
            # [nt, nz, nc, ny, nx] -> [nx, ny, nc, nz, nt]
            return np.transpose(kspace, (4, 3, 2, 1, 0))

        raise ValueError(f"Unknown input_order: {self.input_order}")

    def __getitem__(self, index):
        base_item = self.base_dataset[index]
        kspace = self._extract_kspace(base_item)

        kspace = to_complex_numpy(kspace).astype(np.complex64)
        kspace = self._canonicalize_order(kspace)

        if kspace.ndim != 5:
            raise ValueError(
                f"Expected kspace shape [nx, ny, nc, nz, nt], got {kspace.shape}"
            )

        nx, ny, nc, nz, nt = kspace.shape

        if not (0 <= self.z_index < nz):
            raise IndexError(
                f"z_index={self.z_index} is invalid for nz={nz}."
            )

        # Select one slice/view:
        # [nx, ny, nc, nt]
        kspace_z = kspace[:, :, :, self.z_index, :]

        # Convert Cartesian k-space to coil images:
        # [nx, ny, nc, nt]
        coil_imgs = ifft2c(kspace_z, axes=(0, 1))

        # Reorder to INR expected format:
        # [T, C, H, W]
        img = np.transpose(coil_imgs, (3, 2, 0, 1)).astype(np.complex64)

        if self.crop_square:
            img = center_crop_square(img)

        # Estimate sensitivity maps:
        # [C, H, W]
        smap = estimate_smap_from_rss(img)

        sample = {
            "img": img,
            "smap": smap,
            "z_index": self.z_index,
        }

        if self.keep_original_item and isinstance(base_item, dict):
            for k, v in base_item.items():
                if k != self.kspace_key:
                    sample[k] = v

        if self.return_torch:
            sample["img"] = torch.from_numpy(sample["img"])
            sample["smap"] = torch.from_numpy(sample["smap"])

        return sample


if __name__ == "__main__":
    cds = CineDataset(r"D:\MRI_DATASETS\Test")

    ds = CMRxReconToINRDataset(
        base_dataset=cds,
        kspace_key="kspace_full",
        z_index=0,
        input_order="nxnycnznt",
        crop_square=True,
        return_torch=True,
    )

    x = ds[0]


    import matplotlib.pyplot as plt
    ## plot two images
    fig, axes = plt.subplots(1, 4, figsize=(10, 4))
    for i in range(4):
        axes[i].imshow(x['img'][i].sum(dim=0).abs(), cmap="gray")
    # axes[0].imshow(x['img'][0, 0].real, cmap='gray')
    # axes[0].set_title('Real Component')
    # axes[0].axis('off')
    #
    # axes[1].imshow(np.abs(x['img'][0, 0]), cmap='gray')
    # axes[1].set_title('Image 2 (Magnitude)')
    # axes[1].axis('off')

    plt.tight_layout()
    plt.show()
