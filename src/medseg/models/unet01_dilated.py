import torch.nn as nn
import torch.nn.functional as F

class DilatedNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1, dilation=1)
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1, dilation=2)
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1, dilation=3)
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1, dilation=4)

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.dec_conv0 = nn.ConvTranspose2d(64, 64, 3, padding=1, dilation=4)
        self.dec_conv1 = nn.ConvTranspose2d(64, 64, 3, padding=1, dilation=3)
        self.dec_conv2 = nn.ConvTranspose2d(64, 64, 3, padding=1, dilation=2)
        self.dec_conv3 = nn.ConvTranspose2d(64, 1, 3, padding=1, dilation=1)

    def forward(self, x):
        # encoder
        e0 = (F.relu(self.enc_conv0(x)))
        e1 = (F.relu(self.enc_conv1(e0)))
        e2 = (F.relu(self.enc_conv2(e1)))
        e3 = (F.relu(self.enc_conv3(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        d0 = F.relu(self.dec_conv0(b))
        d1 = F.relu(self.dec_conv1(d0))
        d2 = F.relu(self.dec_conv2(d1))
        d3 = self.dec_conv3(d2)  # no activation
        return d3