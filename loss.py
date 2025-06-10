import torch
import torch.nn as nn
import torch.nn.functional as F

def get_gaussian_kernel(kernel_size: int = 5, sigma: float = 1.0, channels: int = 3):
    """1D Gaussian kernel, extended to 2D separable kernel, for conv2d."""
    # 1D kernel
    ax = torch.arange(kernel_size) - (kernel_size - 1) / 2.0
    gauss = torch.exp(-0.5 * (ax / sigma)**2)
    gauss = gauss / gauss.sum()
    # outer product → 2D kernel
    kernel2d = gauss[:, None] * gauss[None, :]
    # shape [out_channels, in_channels, k, k]
    kernel2d = kernel2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
    return kernel2d

class LaplacianPyramidLoss(nn.Module):
    """
    Computes Laplacian Pyramid Loss between input and target images.
    At each level: Lap = gaussian_blur(I) - upsample( gaussian_blur(downsample(I)) ).
    Loss = sum over levels of L1(lap_in - lap_target).
    """
    def __init__(self, max_levels: int = 3, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        self.max_levels = max_levels
        self.kernel_size = kernel_size
        self.sigma = sigma

    def _gaussian_pyramid(self, x):
        """Build Gaussian pyramid: list of images from level 0 (original) to level N."""
        pyramid = [x]
        channels = x.shape[1]
        kernel = get_gaussian_kernel(self.kernel_size, self.sigma, channels).to(x.device)
        padding = self.kernel_size // 2
        for _ in range(self.max_levels):
            # blur
            x = F.conv2d(pyramid[-1], kernel, padding=padding, groups=channels)
            # downsample by factor 2
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
            pyramid.append(x)
        return pyramid

    def _laplacian_pyramid(self, x):
        """Convert Gaussian pyramid into Laplacian pyramid."""
        gauss_pyr = self._gaussian_pyramid(x)
        lap_pyr = []
        channels = x.shape[1]
        for i in range(self.max_levels):
            # upsample next level back to current spatial size
            next_up = F.interpolate(gauss_pyr[i+1], size=gauss_pyr[i].shape[2:], 
                                    mode='bilinear', align_corners=False)
            lap = gauss_pyr[i] - next_up
            lap_pyr.append(lap)
        # last level: keep the coarsest Gaussian
        lap_pyr.append(gauss_pyr[-1])
        return lap_pyr

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input, target: (B, C, H, W), values in [0,1] or similar
        Returns:
            scalar loss
        """
        lap_in  = self._laplacian_pyramid(input)
        lap_tgt = self._laplacian_pyramid(target)
        loss = 0.0
        for li, lt in zip(lap_in, lap_tgt):
            loss = loss + F.l1_loss(li, lt)
        return loss


class TVLoss(nn.Module):
    """
    Total Variation Loss: encourages spatial smoothness while preserving edges.
    TV(I) = sum |I_{i+1,j} - I_{i,j}| + |I_{i,j+1} - I_{i,j}|.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            scalar loss
        """
        # vertical and horizontal differences
        dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return self.weight * (dh + dw)
