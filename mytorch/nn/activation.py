import numpy as np
from .module import Module
from mytorch.tensor import Tensor

class ReLU(Module):
    def forward(self, x):
        return x.relu()
    
class sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()
    
class SimpleSoftmax(Module):
    def forward(self, x):
        return x.softmax()