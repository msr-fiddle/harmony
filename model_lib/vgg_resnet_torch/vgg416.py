""" VGG416 of vDNN (https://dl.acm.org/doi/10.5555/3195638.3195660) """

import torch

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=64, 
                 batch_norm=False, inplace=True, max_pool=False):
        super(ConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.inplace = inplace
        self.max_pool = max_pool
        
        layers = []   
        conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, torch.nn.BatchNorm2d(out_channels), torch.nn.ReLU(inplace)]
        else:
            layers += [conv2d, torch.nn.ReLU(inplace)]
        if max_pool:
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        
        self.feature = torch.nn.Sequential(*layers)
            
    def forward(self, x):
        return self.feature(x)

class LinearLayer0(torch.nn.Module):
    def __init__(self, inplace=True):
        super(LinearLayer0, self).__init__()
        self.inplace = inplace
        
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(inplace),
            torch.nn.Dropout()
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

class LinearLayer1(torch.nn.Module):
    def __init__(self, inplace=True):
        super(LinearLayer1, self).__init__()
        self.inplace = inplace
        
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace),
            torch.nn.Dropout()
        )
            
    def forward(self, x):
        return self.linear(x)

class LinearLayer2(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(LinearLayer2, self).__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.linear(x)

class VGG416(torch.nn.Module):
    def __init__(self, batch_norm=False, inplace=True, num_classes=1000):
        super(VGG416, self).__init__()
        # build cfg from VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        x = 400
        extra_convs = int(x/5)
        cfg = [64]*(2+extra_convs) + ['M']  \
            + [128]*(2+extra_convs) + ['M'] \
            + [256]*(3+extra_convs) + ['M'] \
            + [512]*(3+extra_convs) + ['M'] \
            + [512]*(3+extra_convs) + ['M']
        # conv layers
        layers = []
        in_channels = 3
        for i, v in enumerate(cfg):
            if v != 'M':
                layers.append(
                    ConvLayer(in_channels=in_channels, out_channels=v, 
                              batch_norm=batch_norm, inplace=inplace, 
                              max_pool=True if i+1 < len(cfg) and cfg[i+1] == 'M' else False) )
                in_channels = v
        # linear layers
        layers.append( LinearLayer0(inplace) )
        layers.append( LinearLayer1(inplace) )
        layers.append( LinearLayer2(num_classes) )
        # put together
        self.all_layers = torch.nn.Sequential(*layers)
        del layers
        # initialize weight
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.all_layers(x)
