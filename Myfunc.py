import torch
import numpy as np
import matrixdot_lut_cython
from torch.autograd.function import Function
from utils import im2col,col2im
# ******************** new conv2d by wxt*******************************************
def make_padding(input, padding):
    if padding == (0, 0):
        #return input
        return input.numpy()
    b, c, h, w = input.shape
    p, q = padding
    result = np.zeros((b, c, h+2*p, w+2*q), dtype=np.float32)
    result[:, :, p:-p, q:-q] = input
    return result

def im2bchwkl(input, ksize, stride=(1, 1), padding=(0, 0), dilation=(1, 1), writeable=False):
    if padding != (0, 0):
        assert not writeable, 'No writable in padding mode.'
        input = make_padding(input, (padding[0], padding[1]))

    isize = input.shape
    istrides = input.strides

    H = (isize[2]-(dilation[0]*(ksize[0]-1)+1))/(stride[0])+1
    W = (isize[3]-(dilation[1]*(ksize[1]-1)+1))/(stride[1])+1
    #assert int(H) == H and int(W) == W, 'conv2d not aligned'
    H = int(H)
    W = int(W)
    istrides = list(istrides+istrides[-2:])
    istrides[2] *= stride[0]
    istrides[3] *= stride[1]
    istrides[4] *= dilation[0]
    istrides[5] *= dilation[1]
    return np.lib.stride_tricks.as_strided(input,
                                           (isize[0], isize[1], H,
                                            W, ksize[0], ksize[1]),
                                           istrides,
                                           writeable=writeable,
                                           )

def matrix_dot_lut(a, b, mul_lut, axes=2):
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    a, b = np.asarray(a), np.asarray(b)
    as_ = a.shape
    nda = a.ndim
    bs = b.shape
    ndb = b.ndim
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
    oldb = [bs[axis] for axis in notin]

    at = a.transpose(newaxes_a).reshape(newshape_a)
    bt = b.transpose(newaxes_b).reshape(newshape_b)
    #change by wxt
    [hat,wat] = at.shape
    [hbt,wbt] = bt.shape
    #fake quant for test
    # at = np.around(at/at.max() * 127 + 128)
    # bt = np.around(bt/bt.max() * 127 + 128)

    res = np.zeros((hat,wbt))
    at = at.astype(int)
    bt = bt.astype(int)
    #res = matrixdot_cython.dot(at, bt)
    res = matrixdot_lut_cython.dot(at, bt, mul_lut)
    # for i in range(0,hat):
    #     for j in range(0,wbt):
    #         temp_res = 0
    #         for k in range(0,wat):
    #             temp_res += at[i][k] * bt[k][j]
    #         res[i][j] = temp_res
    #res = matrixdot_cython.dot(at, bt)
    #origin numpy
    #res = dot(at, bt)
    return res.reshape(olda + oldb)

def batch_conv2d_f(x, kernel, stride, quant_input_scale, quant_weight_scale, mul_lut):
    x = im2bchwkl(x, kernel.shape[-2:], stride)
    kernel = np.transpose(kernel.numpy(),(1,0,2,3))
    mul_res = matrix_dot_lut(x, kernel, mul_lut, [(1, 4, 5), (0, 2, 3)])
    mul_res = mul_res * float(quant_input_scale) * float(quant_weight_scale)
    #return matrix_dot_lut(x, kernel, [(1, 4, 5), (0, 2, 3)]).transpose(0, 3, 1, 2)
    return mul_res.transpose(0, 3, 1, 2)


def newconv2d(input, weight, bias, stride, padding, quant_input_scale, quant_weight_scale, mul_lut):
    input = input.data/quant_input_scale
    weight = weight.data/quant_weight_scale
    input = make_padding(input,padding)
    B, C, iH, iW = input.shape
    #iC, oC, kH, kW = weight.shape
    oC, iC, kH, kW = weight.shape
    assert C == iC, 'Conv2d channels in not equal.'
    conv_np =  np.array(batch_conv2d_f(input, weight, stride, quant_input_scale, quant_weight_scale, mul_lut))
    bias_np  = bias.data.numpy()
    for i in range(0,oC):
        conv_np[:,i,:,:] =  conv_np[:,i,:,:] + bias_np[i]
    #conv_res_np = conv_np + bias_np
    #zzz  = torch.Tensor(np.array(batch_conv2d_f(input, weight, stride)))
    #return zzz
    #return torch.Tensor(conv_np)
    return torch.tensor(conv_np, requires_grad=True)

def newliner(input, weight, bias, quant_input_scale, quant_weight_scale, mul_lut):
    weight = weight.t()
    input = input.data.numpy()/quant_input_scale
    weight = weight.data.numpy()/quant_weight_scale
    input = input.data.numpy()
    weight = weight.data.numpy()
    bias = bias.data.numpy()
    input = input.astype(int)
    weight = weight.astype(int)
    mul_res = matrixdot_lut_cython.dot(input, weight, mul_lut)
    mul_res = mul_res * float(quant_input_scale) * float(quant_weight_scale)
    res = mul_res + bias
    return torch.tensor(res,requires_grad=True)

class NewConv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, quant_input_scale, quant_weight_scale, mul_lut):
        ctx.parameters = (padding, stride)
        ctx.save_for_backward(input, weight, bias)
        return newconv2d(input, weight, bias, stride, padding, quant_input_scale, quant_weight_scale, mul_lut)

    # @staticmethod
    # def backward(ctx, grad_from_upstream):
    #     """
    #     override the backward function. It receives a Tensor containing the gradient of the loss
    #     with respect to the output of the custom forward pass, and calculates the gradients of the loss
    #     with respect to each of the inputs of the custom forward pass.
    #     """
    #     # print('Performing custom backward of MyConv2d_v2')
    #     padding, stride = ctx.parameters
    #     padding = padding[0]
    #     #stride = stride[0]
    #     inX, in_weight, in_bias = ctx.saved_tensors
    #     nOutCh, nInCh, nKnRows, nKnCols = in_weight.shape
    #     nImgSamples, nInCh, nInImgRows, nInImgCols = inX.shape
    #     paddedX = torch.zeros((nImgSamples, nInCh, nInImgRows+2*padding, nInImgCols+2*padding), dtype=inX.dtype)
    #     paddedX[:,:,padding:nInImgRows+padding,padding:nInImgCols+padding] = inX
    #
    #     nImgSamples, nInCh, nPadImgRows, nPadImgCols = paddedX.shape
    #     nOutCh, nInCh, nKnRows, nKnCols = in_weight.shape
    #     nImgSamples, nOutCh, nOutRows, nOutCols = grad_from_upstream.shape
    #
    #     grad_padX = torch.zeros_like(paddedX)
    #     grad_weight = torch.zeros_like(in_weight)
    #     for outCh in range(nOutCh):
    #         for iRow in range(nOutRows):
    #             startRow = iRow * stride[0]
    #             for iCol in range(nOutCols):
    #                 startCol = iCol * stride[1]
    #
    #                 grad_padX[:, :, startRow:startRow + nKnRows, startCol:startCol + nKnCols] += \
    #                     grad_from_upstream[:, outCh, iRow, iCol].reshape(-1, 1, 1, 1) * \
    #                     in_weight[outCh, :, 0:nKnRows, 0:nKnCols]
    #
    #                 grad_weight[outCh, :, 0:nKnRows, 0:nKnCols] += \
    #                     (paddedX[:, :, startRow:startRow + nKnRows, startCol:startCol + nKnCols] * \
    #                      grad_from_upstream[:, outCh, iRow, iCol].reshape(-1, 1, 1, 1)).sum(axis=0)
    #
    #     grad_inputX = grad_padX[:, :, padding:nPadImgRows - padding, padding:nPadImgCols - padding]
    #
    #     if in_bias is not None:
    #         grad_bias = grad_from_upstream.sum(axis=(0, 2, 3))
    #     else:
    #         grad_bias = None
    #     return grad_inputX, grad_weight, grad_bias, None, None, None, None, None
    @staticmethod
    def backward(ctx, grad_from_upstream):
        inX, in_weight, in_bias = ctx.saved_tensors
        if in_bias is not None:
            grad_bias = grad_from_upstream.sum(axis=(0, 2, 3))
        else:
            grad_bias = None
        padding, stride = ctx.parameters
        m, _, _, _ = inX.shape
        nOutCh, nInCh, nKnRows, nKnCols = in_weight.shape
        X_col = im2col(inX, nKnRows, nKnCols, stride[0], padding[0])
        w_col = in_weight.reshape(nOutCh, -1).numpy()
        grad_from_upstream = grad_from_upstream.reshape(grad_from_upstream.shape[0] * grad_from_upstream.shape[1],
                                                        grad_from_upstream.shape[2] * grad_from_upstream.shape[3])
        grad_from_upstream = np.array(np.vsplit(grad_from_upstream, m))
        grad_from_upstream = np.concatenate(grad_from_upstream, axis=-1)
        grad_inputX = w_col.T @ grad_from_upstream
        #grad_inputX = matrixdot_cython.dot(w_col.T , grad_from_upstream)
        grad_weight = grad_from_upstream @ X_col.T
        #grad_weight = matrixdot_cython.dot(grad_from_upstream , X_col.T)
        grad_inputX = col2im(grad_inputX, inX.shape, nKnRows, nKnCols, stride[0], padding[0])
        grad_weight = grad_weight.reshape((grad_weight.shape[0], nInCh, nKnRows, nKnCols))

        return torch.tensor(grad_inputX), torch.tensor(grad_weight), grad_bias, None, None, None, None, None

class NewLiner(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, quant_input_scale, quant_weight_scale, mul_lut):
        ctx.save_for_backward(input, weight, bias, quant_input_scale, quant_weight_scale)
        return newliner(input, weight, bias, quant_input_scale, quant_weight_scale, mul_lut)
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias , quant_input_scale, quant_weight_scale = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None