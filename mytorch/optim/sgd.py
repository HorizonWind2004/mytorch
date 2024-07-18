import numpy as np       
class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0.9, weight_decay=0.0005):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(param.data) for param in self.parameters]
        self.weight_decay = weight_decay

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.requires_grad and param.grad is not None:
                self.velocities[i] = self.momentum * self.velocities[i] + self.lr * param.grad
                param.data -= self.weight_decay * param.data
                param.data -= self.velocities[i]

    def zero_grad(self):
        for param in self.parameters:
            if param.requires_grad:
                param.grad = np.zeros_like(param.data)