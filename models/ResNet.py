import torch
import torch.nn as nn
import os
if torch.cuda.is_available():
    from app_cuda_quant.quantize_cuda_mul import QuantConv2d, QuantLinear
else:
    from quantize import QuantConv2d, QuantLinear

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50"]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, groups=1, base_width=64, dilation=1):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10,
                 zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # END

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, device, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, **kwargs
    )


def resnet34(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, device, **kwargs
    )


def resnet50(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, device, **kwargs
    )


##############################
##########lianghua############
##############################


def conv3x3_quant(in_planes, out_planes, mul_lut, flag, w_bits, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return QuantConv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, mul_lut=mul_lut,
                       stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=True,
                       w_bits=w_bits, flag=flag)


def conv1x1_quant(in_planes, out_planes, mul_lut, flag, w_bits, stride=1):
    """1x1 convolution"""
    return QuantConv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, mul_lut=mul_lut,
                       stride=stride, bias=True, w_bits=w_bits, flag=flag)


class BasicBlock_quant(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, mul_lut, flag, w_bits, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(BasicBlock_quant, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3_quant(inplanes, planes, mul_lut[0], flag, w_bits[0], stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_quant(planes, planes, mul_lut[1], flag, w_bits[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.flag = flag
        #self.mul_lut = mul_lut

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck_quant(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, mul_lut, flag, w_bits, stride=1,
                 downsample=None, groups=1, base_width=64, dilation=1):
        super(Bottleneck_quant, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1_quant(inplanes, width, mul_lut[0], flag, w_bits[0])
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3_quant(width, width, mul_lut[1], flag, w_bits[1], stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1_quant(width, planes * self.expansion, mul_lut[2], flag, w_bits[2])
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        #self.mul_lut = mul_lut
        self.flag = flag

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18_quant(nn.Module):
    def __init__(self, block, layers, mul_lut, flag, w_bits, num_classes=10,
                 zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(ResNet18_quant, self).__init__()
        self.inplanes = 64
        #self.mul_lut = mul_lut
        self.flag = flag
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=True
        )
        # END
        #self.conv1 = nn.Sequential(
        #    conv3x3_quant(3, self.inplanes, mul_lut[0], flag, w_bits=w_bits[0],
        #                 stride=1, groups=groups, dilation=self.dilation)
        #)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #1,2,3,4 quant set
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       mul_lut_sets=[mul_lut[1], mul_lut[2], mul_lut[3], mul_lut[4], mul_lut[4]],
                                       w_bits_sets=[w_bits[1], w_bits[2], w_bits[3], w_bits[4], w_bits[4]])
        #5678,9
        self.layer2 = self._make_layer(
            block, 128, layers[1],
            mul_lut_sets=[mul_lut[5], mul_lut[6], mul_lut[7], mul_lut[8], mul_lut[9]],
            w_bits_sets=[w_bits[5], w_bits[6], w_bits[7], w_bits[8], w_bits[9]],
            stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2],
            mul_lut_sets=[mul_lut[10], mul_lut[11], mul_lut[12], mul_lut[13], mul_lut[14]],
            w_bits_sets=[w_bits[10], w_bits[11], w_bits[12], w_bits[13], w_bits[14]],
            stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3],
            mul_lut_sets=[mul_lut[15], mul_lut[16], mul_lut[17], mul_lut[18], mul_lut[19]],
            w_bits_sets=[w_bits[15], w_bits[16], w_bits[17], w_bits[18], w_bits[19]],
            stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.fc = nn.Sequential(
        #   QuantLinear(512 * block.expansion, num_classes, mul_lut[20], flag=self.flag, w_bits=w_bits[20])
        #)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, QuantConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_quant):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock_quant):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, mul_lut_sets, w_bits_sets, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_quant(self.inplanes, planes * block.expansion,
                              mul_lut_sets[4], self.flag, w_bits_sets[4], stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                [mul_lut_sets[0], mul_lut_sets[1]],
                self.flag,
                [w_bits_sets[0], w_bits_sets[1]],
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    [mul_lut_sets[2], mul_lut_sets[3]],
                    self.flag,
                    [w_bits_sets[2], w_bits_sets[3]],
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        y = self.bn1(out)
        x = self.relu(y)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet18_quant(arch, block, layers, mul_lut, flag, w_bits, pretrained, progress, device):
    model = ResNet18_quant(block, layers, mul_lut, flag, w_bits)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/cifar10_pth/resnet/" + "fullacc_train" + ".pth", map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def resnet18_quant(mul_lut, flag, w_bits, pretrained=False, progress=True, device="cpu"):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        :param w_bits:
        :param flag:
        :param mul_lut:
    """
    return _resnet18_quant(
        "resnet18", BasicBlock_quant, [2, 2, 2, 2], mul_lut, flag, w_bits, pretrained, progress, device
    )


class ResNet50_quant(nn.Module):
    def __init__(self, block, layers, mul_lut, flag, w_bits, num_classes=10,
                 zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(ResNet50_quant, self).__init__()
        self.inplanes = 64
        #self.mul_lut = mul_lut
        self.flag = flag
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=True
        )
        # END
        #self.conv1 = nn.Sequential(
        #    conv3x3_quant1(3, self.inplanes, mul_lut[0], flag, w_bits=32,
        #                 stride=1, groups=groups, dilation=self.dilation)
        #)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #1,2,3,4 quant set
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       mul_lut_sets=[mul_lut[1], mul_lut[2], mul_lut[3], mul_lut[4], mul_lut[4]],
                                       w_bits_sets=[w_bits[1], w_bits[2], w_bits[3], w_bits[4], w_bits[4]])
        #5678,9
        self.layer2 = self._make_layer(
            block, 128, layers[1],
            mul_lut_sets=[mul_lut[5], mul_lut[6], mul_lut[7], mul_lut[8], mul_lut[9]],
            w_bits_sets=[w_bits[5], w_bits[6], w_bits[7], w_bits[8], w_bits[9]],
            stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2],
            mul_lut_sets=[mul_lut[10], mul_lut[11], mul_lut[12], mul_lut[13], mul_lut[14]],
            w_bits_sets=[w_bits[10], w_bits[11], w_bits[12], w_bits[13], w_bits[14]],
            stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3],
            mul_lut_sets=[mul_lut[15], mul_lut[16], mul_lut[17], mul_lut[18], mul_lut[19]],
            w_bits_sets=[w_bits[15], w_bits[16], w_bits[17], w_bits[18], w_bits[19]],
            stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.fc = nn.Sequential(
        #   QuantLinear(512 * block.expansion, num_classes, mul_lut[20], flag=self.flag, w_bits=w_bits[20])
        #)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, QuantConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_quant):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock_quant):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, mul_lut_sets, w_bits_sets, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_quant(self.inplanes, planes * block.expansion,
                              mul_lut_sets[4], self.flag, w_bits_sets[4], stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                [mul_lut_sets[0], mul_lut_sets[1]],
                self.flag,
                [w_bits_sets[0], w_bits_sets[1]],
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    [mul_lut_sets[2], mul_lut_sets[3]],
                    self.flag,
                    [w_bits_sets[2], w_bits_sets[3]],
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        y = self.bn1(out)
        x = self.relu(y)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet50_quant(arch, block, layers, mul_lut, flag, w_bits, pretrained, progress, device):
    model = ResNet18_quant(block, layers, mul_lut, flag, w_bits)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/cifar10_pth/resnet/" + "fullacc_train" + ".pth", map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def resnet50_quant(mul_lut, flag, w_bits, pretrained=False, progress=True, device="cpu"):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        :param w_bits:
        :param flag:
        :param mul_lut:
    """
    return _resnet50_quant(
        "resnet18", Bottleneck_quant, [3, 4, 6, 3], mul_lut, flag, w_bits, pretrained, progress, device
    )

