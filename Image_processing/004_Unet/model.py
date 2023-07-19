import torch
import torch.nn as nn
import torch.nn.functional as F

class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                              out_channels,
                              **kwargs)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               **kwargs)

    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))))
    

class Unet(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 mid_channels=64,
                 nb_classes=2):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.nb_classes = nb_classes

        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.1)

        self.encode = [UnetBlock(3, self.mid_channels, 
                                 kernel_size=3, stride=1, padding=1)]
        for i in range(4):
            self.encode.append(UnetBlock(self.mid_channels*2**i, 
                                         self.mid_channels*2**(i+1), 
                                         kernel_size=3, stride=1, padding=1))
        self.encode = nn.ModuleList(self.encode)

        self.decode_upconv = []
        self.decode_conv = []

        for i in range(4):
            self.decode_upconv.append(nn.ConvTranspose2d(self.mid_channels*2**(i+1),
                                                         self.mid_channels*2**i, 2, 2))
            self.decode_conv.append(UnetBlock(self.mid_channels*2**(i+1),
                                              self.mid_channels*2**i, 
                                              kernel_size=3, stride=1, padding=1))
        self.decode_upconv = nn.ModuleList(self.decode_upconv)
        self.decode_conv = nn.ModuleList(self.decode_conv)

        self.out = nn.Conv2d(self.mid_channels, self.nb_classes, 1, 1, 0)

    
    def forward(self, x):

        self.do1 = self.encode[0](x)
        self.do1_pooled = self.max_pool(self.do1)
        self.do2 = self.encode[1](self.do1_pooled)
        self.do2_pooled = self.max_pool(self.do2)
        self.do3 = self.encode[2](self.do2_pooled)
        self.do3_pooled = self.max_pool(self.do3)
        self.do4 = self.encode[3](self.do3_pooled)
        self.do4_pooled = self.max_pool(self.do4)
        self.do5 = self.encode[4](self.do4_pooled)
        self.do5 = self.dropout(self.do5)

        self.uo4 = self.decode_upconv[3](self.do5)
        self.o4 = torch.cat((self.do4, self.uo4), dim=1)
        self.o4 = self.decode_conv[3](self.o4)
        self.uo3 = self.decode_upconv[2](self.o4)
        self.o3 = torch.cat((self.do3, self.uo3), dim=1)
        self.o3 = self.decode_conv[2](self.o3)
        self.uo2 = self.decode_upconv[1](self.o3)
        self.o2 = torch.cat((self.do2, self.uo2), dim=1)
        self.o2 = self.decode_conv[1](self.o2)
        self.uo1 = self.decode_upconv[0](self.o2)
        self.o1 = torch.cat((self.do1, self.uo1), dim=1)
        self.o1 = self.decode_conv[0](self.o1)

        return self.out(self.o1)
    

if __name__ == '__main__':
    sample = torch.randn((1, 3, 512, 512))
    model = Unet()
    out = model(sample)
    print(out.shape)