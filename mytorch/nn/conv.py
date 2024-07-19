import numpy as np
from mytorch.tensor import Tensor
from mytorch.nn import Module, Parameter
import mytorch.nn as nn

import numpy as np
from mytorch.tensor import Tensor

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        # He initialization
        stddev = np.sqrt(2 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weight = Parameter(Tensor(np.random.randn(out_channels, in_channels, *self.kernel_size) * stddev))
        self.bias = Parameter(Tensor(np.zeros(out_channels)))

    def forward(self, x):
        batch_size, _, height, width = x.data.shape

        out_height = ((height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0]) + 1
        out_width = ((width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1]) + 1

        out = Tensor(np.zeros((batch_size, self.out_channels, out_height, out_width)), requires_grad=x.requires_grad)

        x_padded = np.pad(x.data, [(0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])], mode='constant')

        # Perform convolution
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                w_start = j * self.stride[1]
                h_end = h_start + self.kernel_size[0]
                w_end = w_start + self.kernel_size[1]
                region = x_padded[:, :, h_start:h_end, w_start:w_end]
                out.data[:, :, i, j] = np.tensordot(region, self.weight.data, axes=([1, 2, 3], [1, 2, 3]))

        out.data += self.bias.data.reshape(1, -1, 1, 1)

        def _backward():
            if x.requires_grad:
                dx_padded = np.zeros_like(x_padded)
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride[0]
                        w_start = j * self.stride[1]
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]
                        np.add.at(dx_padded, (slice(None), slice(None), slice(h_start, h_end), slice(w_start, w_end)),
                                  np.tensordot(out.grad[:, :, i, j], self.weight.data, axes=(1, 0)))
                x.grad = dx_padded[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]] if self.padding != (0, 0) else dx_padded

            if self.weight.requires_grad:
                dweight = np.zeros_like(self.weight.data)
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride[0]
                        w_start = j * self.stride[1]
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]
                        region = x_padded[:, :, h_start:h_end, w_start:w_end]
                        dweight += np.tensordot(out.grad[:, :, i, j], region, axes=([0], [0]))
                self.weight.grad = dweight

            if self.bias.requires_grad:
                self.bias.grad = out.grad.sum(axis=(0, 2, 3))

        out._backward = _backward
        out._prev = {x}
            
        return out
    

class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super(AvgPool2d, self).__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size

    def forward(self, x):
        batch_size, num_channels, height, width = x.data.shape
        
        # Calculate output dimensions
        out_height = ((height - self.kernel_size[0]) // self.stride[0]) + 1
        out_width = ((width - self.kernel_size[1]) // self.stride[1]) + 1

        # Initialize the output tensor
        out = Tensor(np.zeros((batch_size, num_channels, out_height, out_width)), requires_grad=x.requires_grad)
                
        # Perform average pooling
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                w_start = j * self.stride[1]
                h_end = h_start + self.kernel_size[0]
                w_end = w_start + self.kernel_size[1]
                region = x.data[:, :, h_start:h_end, w_start:w_end]
                out.data[:, :, i, j] = np.mean(region, axis=(2, 3))
        
        def _backward():
            dx = np.zeros_like(x.data)
            area = self.kernel_size[0] * self.kernel_size[1] 
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * self.stride[0]
                    w_start = j * self.stride[1]
                    h_end = h_start + self.kernel_size[0]
                    w_end = w_start + self.kernel_size[1]
                    dx[:, :, h_start:h_end, w_start:w_end] += (out.grad[:, :, i, j] / area).reshape(batch_size, num_channels, 1, 1)
                
                x.grad = dx

        out._backward = _backward
        out._prev = {x}
        return out

class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(Tensor(np.ones(num_features)))
        self.beta = Parameter(Tensor(np.zeros(num_features)))
        self.running_mean = Parameter(Tensor(np.zeros(num_features)))
        self.running_var = Parameter(Tensor(np.ones(num_features)))
        self.training = False

    def forward(self, x):
        if self.training:
            mean = x.data.mean(axis=(0, 2, 3), keepdims=True)
            var = x.data.var(axis=(0, 2, 3), keepdims=True) + self.eps
            self.running_mean.data = self.momentum * self.running_mean.data + (1 - self.momentum) * mean.flatten()
            self.running_var.data = self.momentum * self.running_var.data + (1 - self.momentum) * var.flatten()
            x_normalized = (x.data - mean) / np.sqrt(var)
        else:
            mean = self.running_mean.data.reshape(1, self.num_features, 1, 1)
            var = self.running_var.data.reshape(1, self.num_features, 1, 1)
            x_normalized = (x.data - mean) / np.sqrt(var + self.eps)

        gamma_reshaped = self.gamma.data.reshape(1, self.num_features, 1, 1)
        beta_reshaped = self.beta.data.reshape(1, self.num_features, 1, 1)

        out = Tensor(gamma_reshaped * x_normalized + beta_reshaped, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                grad_x_normalized = out.grad * self.gamma.data.reshape(1, self.num_features, 1, 1)
                std_var_inv = 1. / np.sqrt(var)
                dx = grad_x_normalized * std_var_inv

                grad_gamma = np.sum(grad_x_normalized * x_normalized, axis=(0, 2, 3))
                grad_beta = np.sum(out.grad, axis=(0, 2, 3))

                self.gamma.grad = grad_gamma
                self.beta.grad = grad_beta
                x.grad = dx

        out._backward = _backward
        out._prev = {x}

        return out

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

# class BatchNorm1d(Module):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1):
#         super(BatchNorm1d, self).__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.gamma = Parameter(Tensor(np.ones(num_features)))
#         self.beta = Parameter(Tensor(np.zeros(num_features)))
#         self.running_mean = np.zeros(num_features)
#         self.running_var = np.ones(num_features)

#     def forward(self, x):
#         if self.training:
#             mean = x.data.mean(axis=(0), keepdims=True)
#             var = x.data.var(axis=(0), keepdims=True) + self.eps
#             self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.flatten()
#             self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.flatten()
#             x_normalized = (x.data - mean) / np.sqrt(var)
#         else:
#             mean = self.running_mean.reshape(1, self.num_features)
#             var = self.running_var.reshape(1, self.num_features)
#             x_normalized = (x.data - mean) / np.sqrt(var + self.eps)

#         gamma_reshaped = self.gamma.data.reshape(1, self.num_features)
#         beta_reshaped = self.beta.data.reshape(1, self.num_features)

#         out = Tensor(gamma_reshaped * x_normalized + beta_reshaped, requires_grad=x.requires_grad)

#         def _backward():
#             if x.requires_grad:
#                 grad_x_normalized = out.grad * self.gamma.data.reshape(1, self.num_features)
#                 std_var_inv = 1. / np.sqrt(var)
#                 dx = grad_x_normalized * std_var_inv

#                 grad_gamma = np.sum(grad_x_normalized * x_normalized, axis=(0))
#                 grad_beta = np.sum(out.grad, axis=(0))

#                 self.gamma.grad = grad_gamma
#                 self.beta.grad = grad_beta
#                 x.grad = dx

#         out._backward = _backward
#         out._prev = {x}

#         return out

#     def train(self):
#         self.training = True

#     def eval(self):
#         self.training = False