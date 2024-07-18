import numpy as np

class Tensor:
    def __init__(self, data ,requires_grad = False):
        if isinstance(data, Tensor):
            self.data = data.data
            self.requires_grad = requires_grad or data.requires_grad
        else:
            self.data = np.array(data, dtype=np.float64)
            self.requires_grad = requires_grad
            
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        topo_order = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo_order.append(v)

        build_topo(self)

        for v in reversed(topo_order):
            v._backward()
            
    
    def __add__(self, other):
        
        if not isinstance(other, Tensor):
            other = Tensor(other)
            
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data , requires_grad = requires_grad)
        
        def _backward():
            # print('+')
            if self.requires_grad:
                if self.grad is None:
                    self.grad = out.grad
                else:
                    self.grad += out.grad
                    
            if other.requires_grad:
                grad = np.zeros_like(other.data)
                if other.data.shape != out.data.shape:
                    axis = tuple(range(out.grad.ndim - other.data.ndim))
                    grad += out.grad.sum(axis=axis)
                else:
                    grad += out.grad
                
                if other.grad is None:
                    other.grad = grad
                else:
                    other.grad += grad

        out._backward = _backward
        out._prev = {self, other}
        return out
    
    def __matmul__(self, other):
        
        assert isinstance(other, Tensor)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data , requires_grad = requires_grad)

        def _backward():
            # print('@')
            if self.requires_grad:
                if self.grad is None:
                    self.grad = out.grad @ other.data.T
                else:
                    self.grad += out.grad @ other.data.T
            if other.requires_grad:
                if other.grad is None:
                    other.grad = self.data.T @ out.grad
                else:
                    other.grad += self.data.T @ out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out
    
    def ln(self): 
        out = Tensor(np.log(self.data + 1e-12), requires_grad=self.requires_grad)
        
        def _backward():
            # print("ln")
            if self.requires_grad:
                if self.grad is None:
                    self.grad = out.grad / (self.data + 1e-12)
                else:
                    self.grad += out.grad / (self.data + 1e-12)
    
        out._backward = _backward
        out._prev = {self}
        return out
    
    def softmax(self):
        exps = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        out = exps / np.sum(exps, axis=-1, keepdims=True)
        out = Tensor(out, requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                grad = out.grad
                out_data = out.data
                grad = grad - np.sum(grad * out_data, axis=-1, keepdims=True) * out_data
                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad

        out._backward = _backward
        out._prev = {self}
        return out
    
    def relu(self):
        
        out = Tensor(np.maximum(0,self.data),requires_grad = self.requires_grad)

        def _backward():
            # print("relu")
            if self.requires_grad:
                if self.grad is None:
                    self.grad = out.grad * (self.data > 0)
                else:
                    self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        out._prev = {self}
        return out
    
    def __mul__(self, other):
        
        if not isinstance(other, Tensor):
            Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data , requires_grad = requires_grad)

        def _backward(): # we only consider the easiest case: z = x * y and x, y are independent.
            # print("mul")
            if self.requires_grad:
                if self.grad is None:
                    self.grad = out.grad * other.data
                else:
                    self.grad += out.grad * other.data
                    
            if other.requires_grad:
                if other.grad is None:
                    other.grad = out.grad * self.data
                else:
                    other.grad += out.grad * self.data

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __neg__(self):
        
        out = Tensor(-self.data, requires_grad = self.requires_grad)
    
        def _backward():
            # print("neg")
            if self.requires_grad:
                if self.grad is None:
                    self.grad = -out.grad
                else:
                    self.grad -= out.grad
    
        out._backward = _backward
        out._prev = {self}
        return out
    
    def __pow__(self, power):
        
        out = Tensor(self.data ** power, requires_grad = self.requires_grad)

        def _backward():
            # print("**")
            if self.requires_grad:
                if self.grad is None:
                    self.grad = out.grad * (power * (self.data ** (power-1)))
                else:
                    self.grad += out.grad * (power * (self.data ** (power-1)))

        out._backward = _backward
        out._prev = {self}
        return out
    
    def __sub__(self, other):
        
        if not isinstance(other,Tensor):
            other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data - other.data , requires_grad = requires_grad)
        
        def _backward():
            # print('-')
            if self.requires_grad:
                if self.grad is None:
                    self.grad = out.grad
                else:
                    self.grad += out.grad
            if other.requires_grad:
                if other.grad is None:
                    other.grad = -out.grad
                else:
                    other.grad -= out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out
    
    def mean(self):
        
        out = Tensor(self.data.mean(), requires_grad=self.requires_grad)

        def _backward():
            # print("mean")
            if self.requires_grad:
                grad = np.ones_like(self.data) / self.data.size
                if self.grad is None:
                    self.grad = grad * out.grad
                else:
                    self.grad += grad * out.grad

        out._backward = _backward
        out._prev = {self}
        return out
    
    def view(self, *shape):
        
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad)
        
        def _backward():
            # print("view")
            if self.requires_grad:
                grad = out.grad.reshape(self.data.shape)
                self.grad = grad
                    
            # print(self.grad)
    
        out._backward = _backward
        out._prev = {self}
        return out
        
    
    def sigmoid(self):
        out = Tensor(1/(1 + np.exp(-self.data)), requires_grad=self.requires_grad)
    
        def _backward():
            # print("sigmoid")
            if self.requires_grad:
                grad = out.data * (1 - out.data)
                if self.grad is None:
                    self.grad = grad * out.grad
                else:
                    self.grad += grad * out.grad
    
        out._backward = _backward
        out._prev = {self}
        return out
    
    
    def __repr__(self):
        return f"Tensor(\ndata = \n{self.data},\nrequires_grad = {self.requires_grad}\n)"
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return Tensor(self.data[idx], requires_grad = self.requires_grad)
    