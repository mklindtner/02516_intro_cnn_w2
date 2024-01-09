import torch.nn as nn
import torch.nn.functional as F
import torch

class UNetSkip(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder 
        self.enc_conv0 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.pool0 = nn.Conv2d(64,64,2,stride=2,padding=0)

        self.enc_conv1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.pool1 = nn.Conv2d(64,64,2,stride=2,padding=0)

        self.enc_conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.pool2 = nn.Conv2d(64,64,2,stride=2,padding=0)


        self.enc_conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.pool3 = nn.Conv2d(64,64,2,stride=2,padding=0)


        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder       
        self.upsample0 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0)
        self.dec_conv0 = nn.Conv2d(64+64, 64, 3, padding=1)
        
        self.upsample1 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0)
        self.dec_conv1 = nn.Conv2d(64+64, 64, 3, padding=1)

        self.upsample2 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0)
        self.dec_conv2 = nn.Conv2d(64+64, 64, 3, padding=1)

        self.upsample3 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0)
        self.dec_conv3 = nn.Conv2d(64+64, 1, 3, padding=1)


    def forward(self, x):
        # encoder
        e0 = (F.relu(self.enc_conv0(x)))
        e01 = self.pool0(e0)
        e1 = (F.relu(self.enc_conv1(e01)))
        e12 = self.pool1(e1)
        e2 = (F.relu(self.enc_conv2(e12)))
        e23 = self.pool2(e2)
        e3 = F.relu(self.enc_conv3(e23))
        e34 = self.pool3(e3)
        
        # bottleneck
        b = F.relu(self.bottleneck_conv(e34))

        # decoder
        bsample = self.upsample0(b)
        
        print(f'bsample: {bsample.shape}\t e3: {e3.shape}')
        d0_s = torch.concat([bsample, e3], dim=1)
        d0 = F.relu(self.dec_conv0(d0_s))
        
        #concat -> upsample 
        d0 = self.upsample1(d0)
        # print(f'd0: {d0.shape}\t e2: {e2.shape}')        
        d1_s = torch.concat([d0, e2], dim=1)
        d1 = F.relu(self.dec_conv1(d1_s))

        # print(f'd1: {d0.shape}\t e1: {e1.shape}')        
        d1 = self.upsample2(d1)
        d2_s = torch.concat([d1,e1],dim=1)
        d2 = F.relu(self.dec_conv2(d2_s)) 

        d2 = self.upsample3(d2)      
        # print(f'd2: {d2.shape}\t e3: {e3.shape}')  
        d3_s = torch.concat([d2,e0],dim=1)
        d3 = self.dec_conv3(d3_s)        
        return d3