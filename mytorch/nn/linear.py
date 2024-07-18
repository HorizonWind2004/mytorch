import numpy as np
from .module import Module
from mytorch.tensor import Tensor

class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad = True)

class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear,self).__init__()
        # self.weight = Parameter(Tensor(np.zeros((in_features, out_features))))
        self.weight = Parameter(Tensor(np.random.randn(in_features, out_features)*0.01))
        self.bias = Parameter(Tensor(np.zeros(out_features)))

    def forward(self, x):
        return x @ self.weight + self.bias