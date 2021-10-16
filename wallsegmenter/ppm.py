import torch
from torch import nn


# Pyramid Pooling Module
class PPM(nn.Module):
    def __init__(
        self, num_class=150, fc_dim=4096, use_softmax=False, pool_scales=(1, 2, 3, 6)
    ):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
            )

        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(
                fc_dim + len(pool_scales) * 512,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1),
        )

    # Forward pass of the PPM architecture
    def forward(self, x, seg_size=None):
        input_size = x.size()
        ppm_out = [x]

        for pool_scale in self.ppm:
            ppm_out.append(
                nn.functional.interpolate(
                    pool_scale(x),
                    (input_size[2], input_size[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=seg_size, mode="bilinear", align_corners=False
            )
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x
