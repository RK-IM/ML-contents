import torch
import torch.nn as nn

architecture_config = [
    # out_channels, kernel_size, stride, padding
    [64, 7, 2, 3],
    "max_pool",
    [192, 3, 1, 1],
    "max_pool",
    [128, 1, 1, 0],
    [256, 3, 1, 1],
    [256, 1, 1, 0],
    [512, 3, 1, 1],
    "max_pool",
    [[256, 1, 1, 0], [512, 3, 1, 1], 4],
    [512, 1, 1, 0],
    [1024, 3, 1, 1],
    "max_pool",
    [[512, 1, 1, 0], [1024, 3, 1, 1], 2],
    [1024, 3, 1, 1],
    [1024, 3, 2, 1],
    [1024, 3, 1, 1],
    [1024, 3, 1, 1]
]

class YoloBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              **kwargs)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.leaky_relu(self.conv(x))


class YoloV1(nn.Module):
    def __init__(self,
                 split_size=7,
                 num_boxes=2,
                 num_classes=20,
                 in_channels=3):
        super().__init__()

        self.architecture = architecture_config

        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.in_channels = in_channels
        self.convs = self._conv_layers(self.architecture)
        self.fcs = self._fc_layers(self.split_size,
                                   self.num_boxes,
                                   self.num_classes)


    def forward(self, x):
        return self.fcs(self.convs(x))


    def _conv_layers(self, architecture):
        in_channels = self.in_channels
        layers = []
        for layer in architecture:
            if layer == 'max_pool':
                layers.append(nn.MaxPool2d(2, 2))

            elif isinstance(layer, list):
                if isinstance(layer[0], int):
                    layers.append(
                        YoloBlock(in_channels=in_channels,
                                  out_channels=layer[0],
                                  kernel_size=layer[1],
                                  stride=layer[2],
                                  padding=layer[3])
                        )
                    in_channels = layer[0]
                else:
                    for _ in range(layer[-1]):
                        for lyr in layer[:-1]:
                            layers.append(
                                YoloBlock(in_channels=in_channels,
                                          out_channels=lyr[0],
                                          kernel_size=lyr[1],
                                          stride=lyr[2],
                                          padding=lyr[3])
                            )
                            in_channels = lyr[0]

        return nn.Sequential(*layers)


    def _fc_layers(self, S, B, C):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (B*5 + C)),
        )
    

def define_optimizer(optimizer_name, *param, **kwargs):
    """
    Define pytorch optimizer associated to the name.

    Args:
        name (str): pytorch optimizer name
    
    Return:
        (torch.optim)
    """
    try:
        optimizer = getattr(torch.optim, optimizer_name)(*param, **kwargs)
    except AttributeError:
        raise NotImplementedError
    
    return optimizer
