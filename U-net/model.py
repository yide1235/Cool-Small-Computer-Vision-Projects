import torch
from torch import nn
import torch.nn.functional as F
 
class twoConvBlock(nn.Module):
   def __init__(self, in_ch, out_ch):
       super(twoConvBlock, self).__init__()
       self.conv = nn.Sequential(
           nn.Conv2d(in_ch, out_ch, 3, padding=1),
           nn.BatchNorm2d(out_ch),
           nn.ReLU(inplace=True),
           nn.Conv2d(out_ch, out_ch, 3, padding=1),
           nn.BatchNorm2d(out_ch),
           nn.ReLU(inplace=True)
       )
 
   def forward(self, x):
       x = self.conv(x)
       return x
 
 
class up(nn.Module):
   def __init__(self, in_ch, out_ch):
       super(up, self).__init__()
       self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
 
   def forward(self, x1, x2):
       x2 = self.up_scale(x2)
 
       diffY = x1.size()[2] - x2.size()[2]
       diffX = x1.size()[3] - x2.size()[3]
 
       x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
       x = torch.cat([x2, x1], dim=1)
       return x
 
 
class downStep(nn.Module):
   def __init__(self, in_ch, out_ch):
       super(downStep, self).__init__()
       self.pool = nn.MaxPool2d(2, stride=2, padding=0)
       self.conv = twoConvBlock(in_ch, out_ch)
 
   def forward(self, x):
       x = self.conv(self.pool(x))
       return x
 
 
class upStep(nn.Module):
   def __init__(self, in_ch, out_ch):
       super(upStep, self).__init__()
       self.up = up(in_ch, out_ch)
      
 
       self.conv = twoConvBlock(in_ch, out_ch)
 
   def forward(self, x1, x2):
       a = self.up(x1, x2)
       x = self.conv(a)
       return x
 
 
class UNet(nn.Module):
   def __init__(self):
       super(UNet, self).__init__()
       self.conv1 = twoConvBlock(1, 64)
       self.down1 = downStep(64, 128)
       self.down2 = downStep(128, 256)
       self.down3 = downStep(256, 512)
       self.down4 = downStep(512, 1024)
       self.up1 = upStep(1024, 512)
       self.up2 = upStep(512, 256)
       self.up3 = upStep(256, 128)
       self.up4 = upStep(128, 64)
       self.last_conv = nn.Conv2d(64, 2, 1)
 
   def forward(self, x):
       x1 = self.conv1(x)
       x2 = self.down1(x1)
       x3 = self.down2(x2)
       x4 = self.down3(x3)
       x5 = self.down4(x4)
       x1_up = self.up1(x4, x5)
       x2_up = self.up2(x3, x1_up)
       x3_up = self.up3(x2, x2_up)
       x4_up = self.up4(x1, x3_up)
       output = self.last_conv(x4_up)
  
       return output
