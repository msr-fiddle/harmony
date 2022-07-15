""" ResNet1K of SublinearMemCost (https://arxiv.org/abs/1604.06174) """

import torch

class ConvLayer(torch.nn.Module):
    def __init__(self, inplace=True):
        super(ConvLayer, self).__init__()
        self.inplace = inplace
        
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=inplace)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, has_downsample=False, inplace=True):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.has_downsample = has_downsample
        self.inplace = inplace 
        #
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * self.expansion),
            ) if has_downsample else None
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, has_downsample=False, inplace=True):
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.has_downsample = has_downsample
        self.inplace = inplace 
        
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes * self.expansion)
        self.relu = torch.nn.ReLU(inplace=inplace)
        self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * self.expansion),
            ) if has_downsample else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class LinearLayer(torch.nn.Module):
    def __init__(self, expansion, num_classes=1000):
        super(LinearLayer, self).__init__()
        self.expansion = expansion
        self.num_classes = num_classes
        
        self.avgpool = torch.nn.AvgPool2d(7, stride=1)
        self.fc = torch.nn.Linear(512 * expansion, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNet(torch.nn.Module):
    def __init__(self, block, layers, inplace=True, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # starting conv-bn-relu-maxpool
        self.conv = ConvLayer(inplace)
        # four residual stages, each stage contains many residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0], inplace=inplace)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, inplace=inplace)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, inplace=inplace)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, inplace=inplace)
        # ending avg-view,size-linear
        self.linear = LinearLayer(block.expansion, num_classes)
        # initialize weight
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, inplace=True):
        has_downsample = False
        if stride != 1 or self.inplanes != planes * block.expansion:
            has_downsample = True

        layers = []
        layers.append(block(self.inplanes, planes, stride, has_downsample, inplace))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, inplace=inplace))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.linear(x)
        return x

def resnet1026(basic_block=True, num_blocks=[128,128,128,128], inplace=True, num_classes=1000):
    """ SublinearMemCost's configurable resnset, where each residual stage is deepened with configuration.
    Note: conv-bn-relu or linear is counted a one layer (downsample branch is not counted). How to count resnet layer? begin+1, basic+2, bottlenck+3, end+1

    Args:
        basic_block (bool): True for BasicBlock, False for Bottleneck
        num_blocks (list of 4 int): number of blocks in four residual stages
        inplace (bool): for Relu
        num_classes : 1000 for imagenet
    """
    assert isinstance(num_blocks, list) and len(num_blocks) == 4
    if basic_block:
        cnt_layers = 1 + sum(num_blocks)*2 + 1
        assert cnt_layers == 1026
        arch = "resnet%d_basic_%d_%d_%d_%d" % (
            cnt_layers, num_blocks[0], num_blocks[1], num_blocks[2], num_blocks[3])
        
        model = ResNet(BasicBlock, num_blocks, inplace=inplace, num_classes=num_classes)        
    else:
        cnt_layers = 1 + sum(num_blocks)*3 + 1
        assert cnt_layers == 1026
        arch = "resnet%d_bottle_%d_%d_%d_%d" % (
            cnt_layers, num_blocks[0], num_blocks[1], num_blocks[2], num_blocks[3])
        
        model = ResNet(Bottleneck, num_blocks, inplace=inplace, num_classes=num_classes)
    return model, arch
