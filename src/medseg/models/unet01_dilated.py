import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch

class DilatedNet(nn.Module):
    def __init__(self):
        super().__init__()        
        DEC_PAD = 0
        DEC_CONVPAD = 0

        # encoder dilation
        self.dil_conv0 = nn.Conv2d(3, 64, 3, padding="same", dilation=1)
        self.dil_conv1 = nn.Conv2d(64, 64, 3, padding="same", dilation=2)
        self.dil_conv3 = nn.Conv2d(64, 64, 3, padding="same", dilation=4)
        
        # bottleneck
        self.bottleneck_conv0 = nn.Conv2d(64, 64, 3, padding=1)

        #Encoder
        self.enc_conv0 = nn.Conv2d(64, 64, 3) #553 -> 551
        self.pool1 = nn.MaxPool2d(2, 2) #551 -> 275

        self.enc_conv1 = nn.Conv2d(64, 64, 3) #275 -> 273
        self.pool2 = nn.MaxPool2d(2, 2) #273 -> 136

        self.enc_conv2 = nn.Conv2d(64, 64, 3) #136 -> 134
        self.pool3 = nn.MaxPool2d(2, 2) #134 -> 67

        self.enc_conv3 = nn.Conv2d(64, 64, 3) #65
        self.pool4 = nn.MaxPool2d(2, 2) #65 -> 32

        self.bottleneck_conv1 = nn.Conv2d(64, 64, 3, padding=0) #32 = 30

        # decoder       
        self.upsample0 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=DEC_PAD) #30*2 = 60
        self.dec_conv0 = nn.Conv2d(64+64, 64, 3, padding=DEC_CONVPAD) #60-2 = 58
        
        self.upsample1 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=DEC_PAD) #58*2 = 116
        self.dec_conv1 = nn.Conv2d(64+64, 64, 3, padding=DEC_CONVPAD) #116-2 = 114

        self.upsample2 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=DEC_PAD) #114*2 = 228
        self.dec_conv2 = nn.Conv2d(64+64, 64, 3, padding=DEC_CONVPAD) #228-2 = 226

        self.upsample3 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=DEC_PAD) #226*2 = 452
        self.dec_conv3 = nn.Conv2d(64+64, 1, 3, padding=DEC_CONVPAD) #450x450

    def forward(self, x):
        # dilationencoder
        e0 = (F.relu(self.dil_conv0(x)))
        e1 = (F.relu(self.dil_conv1(e0)))
        e3 = (F.relu(self.dil_conv3(e1)))
        e4 = (F.relu(self.bottleneck_conv0(e3)))

        #Encoder
        e5 = F.relu(self.enc_conv0(e4))
        e6 = self.pool1(e5)
        e7 = F.relu(self.enc_conv1(e6))
        e8 = self.pool2(e7)
        e9 = F.relu(self.enc_conv2(e8))
        e10 = self.pool3(e9)
        e11 = F.relu(self.enc_conv3(e10))
        e12 = self.pool4(e11)
        b = F.relu(self.bottleneck_conv1(e12))
        
        # decoder
        d_e1 = self.upsample0(b)

        e11 = TF.center_crop(e11, 60)
        print(f'd_e1: {d_e1.shape}\t e11: {e11.shape}')

        d_e2 = torch.cat((d_e1, e11), dim=1)        
        d_e3 = F.relu(self.dec_conv0(d_e2))
        
        d_e4 = self.upsample1(d_e3)
        e9 = TF.center_crop(e9, 116)

        d_e5 = torch.cat((d_e4, e9), dim=1)        
        d_e6 = F.relu(self.dec_conv1(d_e5))

        d_e7 = self.upsample2(d_e6)        
        e7 = TF.center_crop(e7, 228)
        d_e8 = torch.cat((d_e7, e7), dim=1)
        d_e9 = self.dec_conv2(d_e8)

        d_e10 = self.upsample3(d_e9)
        e5 = TF.center_crop(e5, 452)
        d_e11 = torch.cat((d_e10, e5), dim=1)
        
        d_e12 = self.dec_conv3(d_e11)

        return d_e12