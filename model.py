"""CNN that maps a 100x100 source mask to the temperature field after N_ITERS steps."""
import torch
import torch.nn as nn

import config


def conv_bn_relu(in_ch, out_ch, dilation=1):
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]


class ThermalCNN(nn.Module):
    """Small U-Net.

    Input : (B, 100, 100) or (B, 1, 100, 100), source mask (1 = source block, 0 = empty)
    Output: (B, 1, 100, 100), predicted temperature field

    Design notes:
      * Heat spreads ~10 cells in 200 steps, so the network needs a receptive field of
        50+ cells. Two 2x poolings plus dilated convs in the bottleneck give ~70.
      * Zero padding alone cannot tell the network where the cold boundary is (an empty
        input looks the same as padding), so two fixed coordinate channels are appended
        internally. The public interface is still just the 100x100 mask.
    """

    def __init__(self, base_channels=32, grid_size=config.GRID_SIZE):
        super().__init__()
        c = base_channels

        axis = torch.linspace(-1.0, 1.0, grid_size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("coords", torch.stack([yy, xx]).unsqueeze(0))  # (1, 2, H, W)

        self.enc1 = nn.Sequential(*conv_bn_relu(3, c), *conv_bn_relu(c, c))                  # 100x100
        self.enc2 = nn.Sequential(*conv_bn_relu(c, 2 * c), *conv_bn_relu(2 * c, 2 * c))      # 50x50
        self.bottleneck = nn.Sequential(                                                    # 25x25
            *conv_bn_relu(2 * c, 4 * c, dilation=1),
            *conv_bn_relu(4 * c, 4 * c, dilation=2),
            *conv_bn_relu(4 * c, 4 * c, dilation=4),
        )
        self.dec2 = nn.Sequential(*conv_bn_relu(4 * c + 2 * c, 2 * c), *conv_bn_relu(2 * c, 2 * c))
        self.dec1 = nn.Sequential(*conv_bn_relu(2 * c + c, c), *conv_bn_relu(c, c))
        self.head = nn.Conv2d(c, 1, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()
        x = torch.cat([x, self.coords.expand(x.shape[0], -1, -1, -1)], dim=1)

        e1 = self.enc1(x)                                # 100x100
        e2 = self.enc2(self.pool(e1))                    # 50x50
        b = self.bottleneck(self.pool(e2))               # 25x25
        d2 = self.dec2(torch.cat([self.up(b), e2], 1))   # 50x50
        d1 = self.dec1(torch.cat([self.up(d2), e1], 1))  # 100x100
        return self.head(d1)
