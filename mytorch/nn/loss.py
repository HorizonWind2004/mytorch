import numpy as np
from .module import Module
from mytorch.tensor import Tensor

class MSELoss(Module):
    def forward(self, prediction, target):
        return ((prediction - target) ** 2).mean()

class CrossEntropyLoss(Module):
    def forward(self, prediction, target):
        exps = np.exp(prediction.data - np.max(prediction.data, axis=-1, keepdims=True))
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
        N = prediction.data.shape[0]
        self.data = -np.sum(target.data * np.log(probs + 1e-12)) / N
        self.probs = probs
        self.prediction = prediction
        self.target = target
        return self

    def backward(self):
        if self.prediction.requires_grad:
            N = self.prediction.data.shape[0]
            grad = (self.probs - self.target.data) / N
            if self.prediction.grad is None:
                self.prediction.grad = grad
            else:
                self.prediction.grad += grad
            self.prediction.backward()
