import torch.nn as nn
if torch.cuda.is_available():
    from app_cuda_quant.quantize_cuda_mul import QuantConv2d, QuantLinear
else:
    from quantize import QuantConv2d, QuantLinear

'''
modified to fit dataset size
'''
NUM_CLASSES = 10


class AlexNet(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return x


class AlexNet_quant(nn.Module):
    def __init__(self, mul_lut, flag, w_bit, num_classes=NUM_CLASSES):
        super(AlexNet_quant, self).__init__()
        self.mul_lut0 = mul_lut[0]
        self.mul_lut1 = mul_lut[1]
        self.mul_lut2 = mul_lut[2]
        self.mul_lut3 = mul_lut[3]
        self.mul_lut4 = mul_lut[4]
        self.mul_lut5 = mul_lut[5]
        self.mul_lut6 = mul_lut[6]
        self.mul_lut7 = mul_lut[7]

        self.flag = flag
        self.features = nn.Sequential(
            QuantConv2d(3, 64, 3, self.mul_lut0, padding=1, w_bits=w_bit[0], flag=self.flag),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            QuantConv2d(64, 192, 3, self.mul_lut1, padding=1, w_bits=w_bit[1], flag=self.flag),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            QuantConv2d(192, 384, 3, self.mul_lut2, padding=1, w_bits=w_bit[2], flag=self.flag),
            nn.ReLU(inplace=True),
            QuantConv2d(384, 256, 3, self.mul_lut3, padding=1, w_bits=w_bit[3], flag=self.flag),
            nn.ReLU(inplace=True),
            QuantConv2d(256, 256, 3, self.mul_lut4, padding=1, w_bits=w_bit[4], flag=self.flag),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            QuantLinear(256 * 4 * 4, 4096, self.mul_lut5, w_bits=w_bit[5], flag=self.flag),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            QuantLinear(4096, 4096, self.mul_lut6, w_bits=w_bit[6], flag=self.flag),
            nn.ReLU(inplace=True),
            QuantLinear(4096, num_classes, self.mul_lut7, w_bits=w_bit[7], flag=self.flag),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.contiguous().view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return x



