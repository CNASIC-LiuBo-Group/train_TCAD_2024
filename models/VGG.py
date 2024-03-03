import torch.nn as nn
if torch.cuda.is_available():
    from app_cuda_quant.quantize_cuda_mul import QuantConv2d, QuantLinear
else:
    from quantize import QuantConv2d, QuantLinear


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11():
    return VGG('VGG11')


def VGG13():
    return VGG('VGG13')


def VGG16():
    return VGG('VGG16')


def VGG19():
    return VGG('VGG19')

class VGG_quant(nn.Module):
    def __init__(self, vgg_name, mul_lut, flag):
        super(VGG_quant, self).__init__()
        self.mul_lut = mul_lut
        self.flag = flag
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = QuantLinear(512, 10, self.mul_lut, flag=self.flag)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [QuantConv2d(in_channels, x, 3, self.mul_lut, padding=1, flag=self.flag),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def VGG11_quant(mul_lut, flag):
    return VGG_quant('VGG11', mul_lut, flag)


def VGG13_quant(mul_lut, flag):
    return VGG_quant('VGG13', mul_lut, flag)


def VGG16_quant(mul_lut, flag):
    return VGG_quant('VGG16', mul_lut, flag)


def VGG19_quant(mul_lut, flag):
    return VGG_quant('VGG19', mul_lut, flag)