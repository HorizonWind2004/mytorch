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
   
class BatchNorm2d(Module): # 废弃
    def __init__(self, num_features, eps = 1e-05, momentum = 0.1):
        super(BatchNorm2d,self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(Tensor(np.ones(num_features)))
        self.bias = Parameter(Tensor(np.zeros(num_features)))

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x):
        batch_size, num_features, height, width = x.data.shape

        if x.requires_grad:
            mean = x.data.mean(axis=(0,2,3))
            var = x.data.var(axis=(0,2,3))

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x.data - mean.reshape(1,-1,1,1)) / np.sqrt(var.reshape(1,-1,1,1) + self.eps)

        out = Tensor(self.weight.data.reshape(1,-1,1,1) * x_hat + self.bias.data.reshape(1,-1,1,1), requires_grad=True)

        def _backward():
            if self.weight.requires_grad:
                if self.weight.grad is None:
                    self.weight.grad = np.zeros_like(self.weight.data)
                self.weight.grad = (out.grad * x_hat).sum(axis=(0,2,3))
            if self.bias.requires_grad:
                if self.bias.grad is None:
                    self.bias.grad = np.zeros_like(self.bias.data)
                self.bias.grad = out.grad.sum(axis=(0,2,3))

            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                dx_hat = out.grad * self.weight.data.reshape(1,-1,1,1)
                dx = (1/batch_size) * (1/np.sqrt(var.reshape(1,-1,1,1) + self.eps)) * (batch_size*dx_hat - dx_hat.sum(axis=(0,2,3)).reshape(1,-1,1,1) - x_hat * (dx_hat*x_hat).sum(axis=(0,2,3).reshape(1,-1,1,1)))
                x.grad += dx

        out._backward = _backward
        out._prev = {x}

        return out