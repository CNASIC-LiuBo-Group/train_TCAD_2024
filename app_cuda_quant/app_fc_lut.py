import torch
import torch.nn as nn
import cuda_app_layers
import pdb

class linear_app_lut(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, bias, mul, weight_scale=1.0, activation_scale=1.0):
        ctx.save_for_backward(X, weight, bias)

        (m, n) = X.shape
        (k, _) = weight.shape

        if (X.is_cuda == True):

            # Must flatten the weights so CUDA sees them sequentially.
            weight = torch.transpose(weight, 0, 1).flatten()
            out = cuda_app_layers.linear_app_forward(X, weight, bias, m, n, k, weight_scale, activation_scale, mul)

            return out
        else:
            return cpp_layers.linear_forward(X, weight, bias)

            # Manually calculate FC for debug purposes, if needed.
            # output = torch.empty(m, k, device='cpu')
            # for k in range(m):
            #     for l in range(k):
            #         accumulation = 0.0
            #         for j in range(n):
            #             accumulation += X[k, j] * weight[j, l]
            #         output[k, l] = accumulation + bias[l]
            #
            # return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias
