from mytorch.tensor import Tensor
import pickle
import numpy
class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        for name, param in self._parameters.items():
            yield param
        for name, module in self._modules.items():
            yield from module.parameters()

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)
    
    def children(self):
        for name, model in self._modules.items():
            yield model
            
    def state_dict(self):
        state_dict = {}
        for name, module in self._modules.items():
            state_dict[name] = module.state_dict()
        for name, param in self._parameters.items():
            state_dict[name] = param.data
        return state_dict
    
    def load_state_dict(self, state_dict):
        for name, obj in state_dict.items():
            if isinstance(obj, numpy.ndarray):
                self._parameters[name].data = obj
                # print(f'Loading parameter {name}')
            else:
                self._modules[name].load_state_dict(obj)
        
    def save(self, path):
        state_dict = self.state_dict()
        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)