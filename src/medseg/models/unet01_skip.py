import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch
import logging
LOG = logging.getLogger(__name__)



#Image expected: (batch_size, 3, 553, 553)
class UNetSkip(nn.Module):
    def __init__(self):
        super().__init__()
        ENC_PAD = 0
        ENC_PP = 0
        DEC_PAD = 0
        DEC_CONVPAD = 0
        
        # encoder 
        self.enc_conv0 = nn.Conv2d(3, 64, 3, stride=1, padding=ENC_PAD) #553-2 = 551
        self.pool0 = nn.Conv2d(64,64,2,stride=2,padding=ENC_PP) #551/2 = 275

        self.enc_conv1 = nn.Conv2d(64, 64, 3, stride=1, padding=ENC_PAD) #275-2 = 273
        self.pool1 = nn.Conv2d(64,64,2,stride=2,padding=ENC_PP) #273/2 = 136

        self.enc_conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=ENC_PAD) #136-2 = 134
        self.pool2 = nn.Conv2d(64,64,2,stride=2,padding=ENC_PP) #134/2 = 67


        self.enc_conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=ENC_PAD) #67-2 = 65
        self.pool3 = nn.Conv2d(64,64,2,stride=2,padding=ENC_PP) #65/2 = 32


        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=0) #32 = 30

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
        # encoder
        e0 = (F.relu(self.enc_conv0(x)))
        e01 = self.pool0(e0)
        # LOG.debug(f'e01: {e01.shape}')
        e1 = (F.relu(self.enc_conv1(e01)))
        e12 = self.pool1(e1)
        e2 = (F.relu(self.enc_conv2(e12)))
        e23 = self.pool2(e2)
        e3 = F.relu(self.enc_conv3(e23))
        e34 = self.pool3(e3)
        
        # bottleneck
        b = F.relu(self.bottleneck_conv(e34))
        
        # decoder
        # LOG.info(f'b: {b.shape}')
        bsample = self.upsample0(b)
        # LOG.info(f'e3 before: {e3.shape}')  

        e3 = TF.center_crop(e3, 60)
        # LOG.info(f'e3 crop: {e3.shape}\t bsample: {bsample.shape}')
        d0_s = torch.concat([bsample, e3], dim=1) #64x64
        d0 = F.relu(self.dec_conv0(d0_s))
        
        # LOG.info(f'd0: {d0.shape}')
        
        #concat -> upsample 
        d0 = self.upsample1(d0) #
        e2 = TF.center_crop(e2, 116)

        # LOG.info(f'd0 shape: {d0.shape} \t e2 shape: {e2.shape}')
        d1_s = torch.concat([d0, e2], dim=1)
        d1 = F.relu(self.dec_conv1(d1_s))

        d1 = self.upsample2(d1)
        e1 = TF.center_crop(e1, 228)

        d2_s = torch.concat([d1,e1],dim=1)
        d2 = F.relu(self.dec_conv2(d2_s)) 

        d2 = self.upsample3(d2)      
        e0 = TF.center_crop(e0, 452)
        
        d3_s = torch.concat([d2,e0],dim=1)
        d3 = self.dec_conv3(d3_s)        
        return d3