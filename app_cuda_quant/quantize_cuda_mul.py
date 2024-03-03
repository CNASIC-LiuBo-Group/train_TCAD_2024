import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Function
import numpy as np
from op import *
from app_conv_lut import *
from app_fc_lut import *


# ********************* observers(统计min/max) *********************
class ObserverBase(nn.Module):
    def __init__(self, q_level, bbit):
        super(ObserverBase, self).__init__()
        self.q_level = q_level
        self.bbit = bbit

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        #####WXT修改#####################################################
        if self.q_level == 'L':  # layer级(activation/weight)
            min_old = torch.min(input)
            max_old = torch.max(input)
        elif self.q_level == 'C':  # channel级(conv_weight)
            input = torch.flatten(input, start_dim=1)
            min_old = torch.min(input, 1)[0]
            max_old = torch.max(input, 1)[0]
        elif self.q_level == 'FC':  # channel级(fc_weight)
            min_old = torch.min(input, 1, keepdim=True)[0]
            max_old = torch.max(input, 1, keepdim=True)[0]
        max_max = max(abs(max_old), abs(min_old))
        n_bit = 1 + np.ceil(np.log2(max_max.cpu()))  # 求数据位宽
        v_max = (pow(2, self.bbit - 1) - 1) * pow(2, -(self.bbit - n_bit))
        max_val = v_max
        min_val = -v_max
        self.update_range(min_val, max_val)
        ####原版#########################################################
        # if self.q_level == 'L':     # layer级(activation/weight)
        # min_val = torch.min(input)
        # max_val = torch.max(input)
        # elif self.q_level == 'C':   # channel级(conv_weight)
        # input = torch.flatten(input, start_dim=1)
        # min_val = torch.min(input, 1)[0]
        # max_val = torch.max(input, 1)[0]
        # elif self.q_level == 'FC':  # channel级(fc_weight)
        # min_val = torch.min(input, 1, keepdim=True)[0]
        # max_val = torch.max(input, 1, keepdim=True)[0]

        # self.update_range(min_val, max_val)


class MinMaxObserver(ObserverBase):
    def __init__(self, q_level, bbit, out_channels):
        super(MinMaxObserver, self).__init__(q_level, bbit)
        self.num_flag = 0
        self.out_channels = out_channels
        if self.q_level == 'L':
            self.register_buffer('min_val', torch.zeros((1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((1), dtype=torch.float32))
        elif self.q_level == 'C':
            self.register_buffer('min_val', torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32))
        elif self.q_level == 'FC':
            self.register_buffer('min_val', torch.zeros((out_channels, 1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((out_channels, 1), dtype=torch.float32))

    def update_range(self, min_val_cur, max_val_cur):
        if self.q_level == 'C':
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


class MovingAverageMinMaxObserver(ObserverBase):
    def __init__(self, q_level, out_channels, bbit, momentum=0.1):
        super(MovingAverageMinMaxObserver, self).__init__(q_level, bbit)
        self.momentum = momentum
        self.num_flag = 0
        self.out_channels = out_channels
        if self.q_level == 'L':
            self.register_buffer('min_val', torch.zeros((1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((1), dtype=torch.float32))
        elif self.q_level == 'C':
            self.register_buffer('min_val', torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32))
        elif self.q_level == 'FC':
            self.register_buffer('min_val', torch.zeros((out_channels, 1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((out_channels, 1), dtype=torch.float32))

    def update_range(self, min_val_cur, max_val_cur):
        if self.q_level == 'C':
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


class HistogramObserver(nn.Module):
    def __init__(self, q_level, momentum=0.1, percentile=0.9999):
        super(HistogramObserver, self).__init__()
        self.q_level = q_level
        self.momentum = momentum
        self.percentile = percentile
        self.num_flag = 0
        self.register_buffer('min_val', torch.zeros((1), dtype=torch.float32))
        self.register_buffer('max_val', torch.zeros((1), dtype=torch.float32))

    @torch.no_grad()
    def forward(self, input):
        # MovingAveragePercentileCalibrator
        # PercentileCalibrator
        max_val_cur = torch.kthvalue(input.abs().view(-1), int(self.percentile * input.view(-1).size(0)), dim=0)[0]
        # MovingAverage
        if self.num_flag == 0:
            self.num_flag += 1
            max_val = max_val_cur
        else:
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
        self.max_val.copy_(max_val)


# ********************* quantizers（量化器，量化） *********************
# 取整(饱和/截断ste)
class Round(Function):
    @staticmethod
    def forward(self, input, observer_min_val, observer_max_val, q_type):
        # 对称
        if q_type == 0:
            max_val = torch.max(torch.abs(observer_min_val), torch.abs(observer_max_val))
            min_val = -max_val
        # 非对称
        else:
            max_val = observer_max_val
            min_val = observer_min_val
        self.save_for_backward(input, min_val, max_val)
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, min_val, max_val = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.gt(max_val)] = 0
        grad_input[input.lt(min_val)] = 0
        return grad_input, None, None, None


class Quantizer(nn.Module):
    def __init__(self, bits, observer, activation_weight_flag, qaft=False, union=False, flag=0):
        super(Quantizer, self).__init__()
        self.bits = bits
        self.observer = observer
        self.activation_weight_flag = activation_weight_flag
        self.qaft = qaft
        self.union = union
        self.q_type = 0
        self.flag = flag
        # scale/zero_point/eps
        if self.observer.q_level == 'L':
            self.register_buffer('scale', torch.ones((1), dtype=torch.float32))
            self.register_buffer('zero_point', torch.zeros((1), dtype=torch.float32))
        elif self.observer.q_level == 'C':
            self.register_buffer('scale', torch.ones((self.observer.out_channels, 1, 1, 1), dtype=torch.float32))
            self.register_buffer('zero_point', torch.zeros((self.observer.out_channels, 1, 1, 1), dtype=torch.float32))
        elif self.observer.q_level == 'FC':
            self.register_buffer('scale', torch.ones((self.observer.out_channels, 1), dtype=torch.float32))
            self.register_buffer('zero_point', torch.zeros((self.observer.out_channels, 1), dtype=torch.float32))
        self.register_buffer('eps', torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32))

    def update_qparams(self):
        raise NotImplementedError

    # 取整(ste)
    def round(self, input, observer_min_val, observer_max_val, q_type):
        output = Round.apply(input, observer_min_val, observer_max_val, q_type)
        return output

    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.bits != 1
        else:
            if not self.qaft:
                # qat, update quant_para
                if self.training:
                    if not self.union:
                        self.observer(input)  # update observer_min and observer_max
                    self.update_qparams()  # update scale and zero_point
            if self.flag:  # 近似乘法
                output = (torch.clamp(self.round(input / self.scale.clone() - self.zero_point,
                                                 self.observer.min_val / self.scale - self.zero_point,
                                                 self.observer.max_val / self.scale - self.zero_point, self.q_type),
                                      self.quant_min_val, self.quant_max_val) + self.zero_point) * self.scale.clone()
            else:  # 非近似乘法
                output = (torch.clamp(self.round(input / self.scale.clone() - self.zero_point,
                                                 self.observer.min_val / self.scale - self.zero_point,
                                                 self.observer.max_val / self.scale - self.zero_point, self.q_type),
                                      self.quant_min_val, self.quant_max_val) + self.zero_point) * self.scale.clone()

            # output = (torch.clamp(self.round(input / self.scale.clone() - self.zero_point,
            #                                  self.observer.min_val / self.scale - self.zero_point,
            #                                  self.observer.max_val / self.scale - self.zero_point, self.q_type),
            #                       self.quant_min_val, self.quant_max_val) + self.zero_point)
        # return output
        return output, self.scale.clone()


'''
class SignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(SignedQuantizer, self).__init__(*args, **kwargs)
        if self.activation_weight_flag == 0:  # weight
            self.register_buffer('quant_min_val', torch.tensor((-((1 << (self.bits - 1)) - 1)), dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor(((1 << (self.bits - 1)) - 1), dtype=torch.float32))
        elif self.activation_weight_flag == 1:  # activation
            self.register_buffer('quant_min_val', torch.tensor((-(1 << (self.bits - 1))), dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor(((1 << (self.bits - 1)) - 1), dtype=torch.float32))
        else:
            print('activation_weight_flag error')


class UnsignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(UnsignedQuantizer, self).__init__(*args, **kwargs)
        if self.activation_weight_flag == 0:  # weight
            self.register_buffer('quant_min_val', torch.tensor((0), dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor(((1 << self.bits) - 2), dtype=torch.float32))
        elif self.activation_weight_flag == 1:  # activation
            self.register_buffer('quant_min_val', torch.tensor((0), dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor(((1 << self.bits) - 1), dtype=torch.float32))
        else:
            print('activation_weight_flag error')
'''


# 对称量化
class SymmetricQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(SymmetricQuantizer, self).__init__(*args, **kwargs)
        if self.activation_weight_flag == 0:  # weight
            self.register_buffer('quant_min_val', torch.tensor((-((1 << (self.bits - 1)) - 1)), dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor(((1 << (self.bits - 1)) - 1), dtype=torch.float32))
        elif self.activation_weight_flag == 1:  # activation
            self.register_buffer('quant_min_val', torch.tensor((-(1 << (self.bits - 1))), dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor(((1 << (self.bits - 1)) - 1), dtype=torch.float32))
        else:
            print('activation_weight_flag error')
        self.q_type = 0


    def update_qparams(self):
        # self.q_type = 0
        # quant_range = float(self.quant_max_val - self.quant_min_val) / 2                                # quantized_range
        # quant_range = self.quant_max_val   # quantized_range
        # float_range = torch.max(torch.abs(self.observer.min_val), torch.abs(self.observer.max_val))     # float_range
        quant_range = pow(2, self.bits - 1) - 1
        float_range = torch.abs(self.observer.max_val)  # float_range
        scale = float_range / quant_range  # scale
        scale = torch.max(scale, self.eps)  # processing for very small scale
        zero_point = torch.zeros_like(scale)  # zero_point
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)


# 非对称量化
class AsymmetricQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(AsymmetricQuantizer, self).__init__(*args, **kwargs)
        if self.activation_weight_flag == 0:  # weight
            self.register_buffer('quant_min_val', torch.tensor((0), dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor(((1 << self.bits) - 2), dtype=torch.float32))
        elif self.activation_weight_flag == 1:  # activation
            self.register_buffer('quant_min_val', torch.tensor((0), dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor(((1 << self.bits) - 1), dtype=torch.float32))
        else:
            print('activation_weight_flag error')
        self.q_type = 1

    def update_qparams(self):
        # self.q_type = 1
        quant_range = float(self.quant_max_val - self.quant_min_val)  # quantized_range
        float_range = self.observer.max_val - self.observer.min_val  # float_range
        scale = float_range / quant_range  # scale
        scale = torch.max(scale, self.eps)  # processing for very small scale
        sign = torch.sign(self.observer.min_val)
        zero_point = sign * torch.floor(torch.abs(self.observer.min_val / scale) + 0.5)  # zero_point
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)


# ********************* 量化卷积（同时量化A/W，并做卷积） *********************
class QuantConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 mul_lut,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 a_bits=8,
                 w_bits=8,
                 q_type=0,
                 q_level=1,
                 weight_observer=1,
                 quant_inference=False,
                 qaft=False,
                 ptq=False,
                 percentile=0.9999,
                 flag_quant=1,
                 flag_app=1
                 ):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.kernel_size = (kernel_size, kernel_size)
        self.kernal_size_number = kernel_size * kernel_size
        self.out_channels = out_channels
        self.dilation = (dilation, dilation)
        self.padding = (padding, padding)
        self.stride = (stride, stride)
        self.in_channels = in_channels
        self.weight = nn.Parameter(torch.rand(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.bias = nn.Parameter(torch.rand(self.out_channels))

        self.quant_inference = quant_inference
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mul_lut = (torch.from_numpy(mul_lut)).to(self.device)
        self.flag_quant = flag_quant
        self.flag_app = flag_app
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                    q_level='L', out_channels=None, bbit=a_bits), activation_weight_flag=1, qaft=qaft, flag=flag_app)
                if weight_observer == 0:
                    if q_level == 0:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                            q_level='C', out_channels=out_channels, bbit=w_bits), activation_weight_flag=0, qaft=qaft,
                                                                   flag=flag_app)
                    else:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                            q_level='L', out_channels=None, bbit=w_bits), activation_weight_flag=0, qaft=qaft,
                                                                   flag=flag_app)
                else:
                    if q_level == 0:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                            q_level='C', out_channels=out_channels, bbit=w_bits), activation_weight_flag=0, qaft=qaft,
                                                                   flag=flag_app)
                    else:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                            q_level='L', out_channels=None, bbit=w_bits), activation_weight_flag=0, qaft=qaft,
                                                                   flag=flag_app)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                    q_level='L', out_channels=None, bbit=a_bits), activation_weight_flag=1, qaft=qaft)
                if weight_observer == 0:
                    if q_level == 0:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                            q_level='C', out_channels=out_channels, bbit=w_bits), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                            q_level='L', out_channels=None, bbit=w_bits), activation_weight_flag=0, qaft=qaft)
                else:
                    if q_level == 0:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                            q_level='C', out_channels=out_channels, bbit=w_bits), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                            q_level='L', out_channels=None, bbit=w_bits), activation_weight_flag=0, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(
                q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft, flag=flag_app)
            if weight_observer == 0:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                        q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft, flag=flag_app)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                        q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft, flag=flag_app)
            else:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                        q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft, flag=flag_app)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                        q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft, flag=flag_app)

    def forward(self, input):

        if self.flag_quant:
            quant_input, quant_input_scale = self.activation_quantizer(input)
            # if not self.quant_inference:
            #     quant_weight, quant_weight_scale = self.weight_quantizer(self.weight)
            quant_weight, quant_weight_scale = self.weight_quantizer(self.weight)
            # else:
            #     quant_weight = self.weight
            # output = F.conv2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.dilation,
            #                    self.groups)

            if self.flag_app:
                output = convAppx_app_lut.apply(quant_input, quant_weight, 
						self.bias, self.mul_lut, self.padding, self.stride, quant_weight_scale, quant_input_scale)
            else:
                output = F.conv2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.dilation,
                                  self.groups)
        else:
            output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation,
                              self.groups)
            '''
        output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation,
                          self.groups)'''
        return output


class QuantLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 mul_lut,
                 bias=True,
                 a_bits=8,
                 w_bits=8,
                 q_type=0,
                 q_level=1,
                 weight_observer=1,
                 quant_inference=False,
                 qaft=False,
                 ptq=False,
                 percentile=0.9999,
                 flag_quant=1,
                 flag_app=1
                 ):
        super(QuantLinear, self).__init__(in_features, out_features, bias)

        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))


        self.quant_inference = quant_inference
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mul_lut = (torch.from_numpy(mul_lut)).to(self.device)
        self.flag_quant = flag_quant
        self.flag_app = flag_app
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                    q_level='L', out_channels=None, bbit=a_bits), activation_weight_flag=1, qaft=qaft, flag=flag_app)
                if weight_observer == 0:
                    if q_level == 0:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                            q_level='FC', out_channels=out_features, bbit=w_bits), activation_weight_flag=0, qaft=qaft,
                                                                   flag=flag_app)
                    else:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                            q_level='L', out_channels=None, bbit=w_bits), activation_weight_flag=0, qaft=qaft,
                                                                   flag=flag_app)
                else:
                    if q_level == 0:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                            q_level='FC', out_channels=out_features, bbit=w_bits), activation_weight_flag=0, qaft=qaft,
                                                                   flag=flag_app)
                    else:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                            q_level='L', out_channels=None, bbit=w_bits), activation_weight_flag=0, qaft=qaft,
                                                                   flag=flag_app)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                    q_level='L', out_channels=None, bbit=a_bits), activation_weight_flag=1, qaft=qaft)
                if weight_observer == 0:
                    if q_level == 0:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                            q_level='FC', out_channels=out_features, bbit=w_bits), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                            q_level='L', out_channels=None, bbit=w_bits), activation_weight_flag=0, qaft=qaft)
                else:
                    if q_level == 0:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                            q_level='FC', out_channels=out_features, bbit=w_bits), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                            q_level='L', out_channels=None, bbit=w_bits), activation_weight_flag=0, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(
                q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft, flag=flag_app)
            if weight_observer == 0:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                        q_level='FC', out_channels=out_features), activation_weight_flag=0, qaft=qaft, flag=flag_app)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                        q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft, flag=flag_app)
            else:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                        q_level='FC', out_channels=out_features), activation_weight_flag=0, qaft=qaft, flag=flag_app)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                        q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft, flag=flag_app)

    def forward(self, input):
        if self.flag_quant:
            quant_input, quant_input_scale = self.activation_quantizer(input)
            quant_weight, quant_weight_scale = self.weight_quantizer(self.weight)
            if self.flag_app:
            # output = newliner(quant_input, quant_weight, self.bias, quant_input_scale, quant_weight_scale, self.mul_lut)
                output = linear_app_lut.apply(quant_input, quant_weight, self.bias, self.mul_lut, quant_weight_scale, quant_input_scale)
            else:
                output = F.linear(quant_input, quant_weight, self.bias)
        # output = F.linear(quant_input, quant_weight, self.bias)
        else:
            output = F.linear(input, self.weight, self.bias)

        return output


class QuantReLU(nn.ReLU):
    def __init__(self, inplace=False, a_bits=8, q_type=0, qaft=False, ptq=False, percentile=0.9999):
        super(QuantReLU, self).__init__(inplace)
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                    q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                    q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(
                q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.relu(quant_input, self.inplace)
        return output


class QuantLeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope=0.01, inplace=False, a_bits=8, q_type=0, qaft=False,
                 ptq=False, percentile=0.9999):
        super(QuantLeakyReLU, self).__init__(negative_slope, inplace)
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                    q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                    q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(
                q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.leaky_relu(quant_input, self.negative_slope, self.inplace)
        return output


class QuantSigmoid(nn.Sigmoid):
    def __init__(self, a_bits=8, q_type=0, qaft=False, ptq=False, percentile=0.9999):
        super(QuantSigmoid, self).__init__()
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                    q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                    q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(
                q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.sigmoid(quant_input)
        return output
