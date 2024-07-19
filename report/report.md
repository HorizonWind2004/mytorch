<div style="height: 100pt;">
</div>

<div style="style=display: block; margin-left: auto; margin-right: auto; width: 60%; height: auto;">
  <img src="logo.bmp">
  <br>
  <br>
</div>

<div style="height: 40pt;">
</div>

<div style="text-align:center;font-size:20pt;">
    <strong>暑期深度学习实验报告</strong><br>
    <br>
</div>

<div style="height: 80pt;">
</div>

<div style="display: flex; align-items: center;justify-content: center;font-size:14pt;">
  <div style="display:flex; align-items: center; width: 70pt; background-color: rgba(255, 255, 255, 0);justify-content: center;">
实验名称<span style="margin-right: 7pt">:</span>
</div>
<div style="display:flex; align-items: center; width: 200pt; background-color: rgba(255, 255, 255, 0);justify-content: center;border-bottom: 1pt solid #000;">
MNIST 数字图像分类
</div>
</div>

<div style="height: 7pt;">
</div>

<div style="display: flex; align-items: center;justify-content: center;font-size:14pt;">
  <div style="display:flex; align-items: center; width: 70pt; background-color: rgba(255, 255, 255, 0);justify-content: center;">
小组成员</span><span style="margin-right: 7pt">:</span>
</div>
<div style="display:flex; align-items: center; width: 200pt; background-color: rgba(255, 255, 255, 0);justify-content: center;border-bottom: 1pt solid #000;">
谢<span style="margin-left: 14pt;"></span>集 322010xxxx
</div>
</div>

<div style="height: 7pt;">
</div>

<div style="display: flex; align-items: center;justify-content: center;font-size:14pt;">
  <div style="display:flex; align-items: center; width: 70pt; background-color: rgba(255, 255, 255, 0);justify-content: center;">
<span style="margin-left: 70pt;"></span>
</div>
<div style="display:flex; align-items: center; width: 200pt; background-color: rgba(255, 255, 255, 0);justify-content: center;border-bottom: 1pt solid #000;">
李浩浩 3220105930
</div>
</div>

<div style="height: 7pt;">
</div>

<div style="display: flex; align-items: center;justify-content: center;font-size:14pt;">
  <div style="display:flex; align-items: center; width: 70pt; background-color: rgba(255, 255, 255, 0);justify-content: center;">
<span style="margin-left: 70pt;"></span>
</div>
<div style="display:flex; align-items: center; width: 200pt; background-color: rgba(255, 255, 255, 0);justify-content: center;border-bottom: 1pt solid #000;">
赵一帆 322010xxxx
</div>
</div>

<div style="height: 7pt;">
</div>

<div style="display: flex; align-items: center;justify-content: center;font-size:14pt;">
  <div style="display:flex; align-items: center; width: 70pt; background-color: rgba(255, 255, 255, 0);justify-content: center;">
<span style="margin-left: 70pt;"></span>
</div>
<div style="display:flex; align-items: center; width: 200pt; background-color: rgba(255, 255, 255, 0);justify-content: center;border-bottom: 1pt solid #000;">
杨正宇 322010xxxx
</div>
</div>

<div style="height: 7pt;">
</div>

<div style="display: flex; align-items: center;justify-content: center;font-size:14pt;">
  <div style="display:flex; align-items: center; width: 70pt; background-color: rgba(255, 255, 255, 0);justify-content: center;">
课程名称<span style="margin-right: 7pt">:</span>
</div>
<div style="display:flex; align-items: center; width: 200pt; background-color: rgba(255, 255, 255, 0);justify-content: center;border-bottom: 1pt solid #000;">
课程综合实践 I
</div>
</div>

<div style="height: 7pt;">
</div>

<div style="display: flex; align-items: center;justify-content: center;font-size:14pt;">
  <div style="display:flex; align-items: center; width: 70pt; background-color: rgba(255, 255, 255, 0);justify-content: center;">
指导老师<span style="margin-right: 7pt">:</span>
</div>
<div style="display:flex; align-items: center; width: 200pt; background-color: rgba(255, 255, 255, 0);justify-content: center;border-bottom: 1pt solid #000;">
杨<span style="margin-left: 14pt;"></span>易，范鹤鹤
</div>
</div>

<div style="height: 7pt;">
</div>

<div style="page-break-before: always;"></div>

<div style="height: 14pt;">
</div>
<hr></hr>

# Part I: 基于PyTorch实现

# Part II: 基于NumPy实现
## 1. mytorch 框架
我们希望通过此次试验，对PyTorch框架的工作原理有更深入的理解，因此我们将手搓一个简化版的torch框架：mytorch。虽然torch的底层实现我们目前难以模仿，但我们通过模仿它的接口设计，在此次分类任务中，基本保证了用户可以在torch与mytorch之间无痛切换。
### 1.1 框架结构
```md
mytorch
│  tensor.py
│  __init__.py
│
├─nn
│  │  activation.py
│  │  conv.py
│  │  linear.py
│  │  loss.py
│  │  module.py
│  │  __init__.py
│
└─optim
   │  sgd.py
   │  __init__.py
```

### 1.2 张量（Tensor）

#### 简介
我在实验中实现的自定义深度学习框架中的核心类——`Tensor`类。该类实现了基本的张量操作以及自动求导功能。下面将详细解释该类的设计、实现及其功能。

#### `Tensor` 类的设计与实现

`Tensor` 类是该深度学习框架的核心组件，负责存储数据、跟踪计算图和自动求导。下面详细解释该类的每个部分。

##### 1. 类初始化与基本属性

```python
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
```
- `data`：张量的数据，可以是另一个`Tensor`对象或可转换为NumPy数组的其他数据类型。
- `requires_grad`：布尔值，指示是否需要计算梯度。
- `grad`：存储梯度的张量，初始值为`None`。
- `_backward`：存储反向传播函数。
- `_prev`：存储计算图中前驱节点的集合。

##### 2. 自动求导功能

```python
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
```
- `backward` 方法实现了自动求导功能。首先，如果当前张量的`grad`属性为`None`，则初始化为全1张量。然后，通过拓扑排序算法，将计算图中所有节点按拓扑顺序排列。然后dfs遍历计算图，对于每个节点，调用其`_backward`方法，该方法根据链式法则计算梯度。

##### 3. 基本运算符重载

- **加法**：
```python
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
    
```

直接将两个张量的`data`属性相加，并将`requires_grad`属性设置为两个张量中任意一个的`requires_grad`属性。反向传播函数根据链式法则计算梯度，不妨设损失函数为 $\mathcal L$ ，对于加法的部分显然有：

$$
\frac{\partial \mathcal L}{\partial x_0} = \frac{\partial \mathcal L}{\partial y} \frac{\partial y}{\partial x_0}
$$ 

其中：

$$
y = x_0 + x_1
$$ 

所以：

$$
\frac{\partial \mathcal L}{\partial x_0} = \frac{\partial \mathcal L}{\partial y} \frac{\partial y}{\partial x_0} = \frac{\partial \mathcal L}{\partial y}
$$ 

我们只需要将 $\frac{\partial \mathcal L}{\partial y}$ 加给 `self.grad` 属性即可，也就是将 `out.grad` 加给 `self.grad` 属性。


- **减法**：
```python
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
```

减法和加法是同理的，这里不再赘述。

- **矩阵乘法**：
```python
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
```

同理，我们可以将两个张量的`data`属性相乘，并将`requires_grad`属性设置为两个张量中任意一个的`requires_grad`属性。反向传播函数根据链式法则计算梯度，不妨设损失函数为 $\mathcal L$ ，对于矩阵乘法的部分显然有：

$$
\frac{\partial \mathcal L}{\partial x_0} = \frac{\partial \mathcal L}{\partial y} \frac{\partial y}{\partial x_0} = \frac{\partial \mathcal L}{\partial y} \times x_1^T
$$ 

我们只需要将 $\frac{\partial \mathcal L}{\partial y} \times x_1^T$ 加给 `self.grad` 属性即可，也就是将 `out.grad` 乘以 `other.data.T` 加给 `self.grad` 属性。

- **乘法**：
```python
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
```

同理，我们可以将两个张量的`data`属性相乘，并将`requires_grad`属性设置为两个张量中任意一个的`requires_grad`属性。反向传播函数根据链式法则计算梯度，不妨设损失函数为 $\mathcal L$ ，对于乘法的部分显然有：

$$
\frac{\partial \mathcal L}{\partial x_0} = \frac{\partial \mathcal L}{\partial y} \frac{\partial y}{\partial x_0} = \frac{\partial \mathcal L}{\partial y} \times x_1
$$ 

我们只需要将 $\frac{\partial \mathcal L}{\partial y} \times x_1$ 加给 `self.grad` 属性即可，也就是将 `out.grad` 乘以 `other.data` 加给 `self.grad` 属性。

我们还需要实现 `__rmul__` 方法，使得`x * y` 等价于 `y * x`。

- **负号**：
```python
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
```

这个部分相当简单，只需要将 `data` 属性取负即可，梯度的部分也是直接取负。

- **幂运算**：
```python
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
```

只需要计算 `data` 属性的幂即可，梯度的部分根据公式，我们可以计算得到：


$$
\frac{\partial \mathcal L}{\partial x_0} = \frac{\partial \mathcal L}{\partial y} \frac{\partial y}{\partial x_0} = \frac{\partial \mathcal L}{\partial y} \times x_0^{\text{power}-1}\cdot \text{power}
$$ 

我们只需要将 $\frac{\partial \mathcal L}{\partial y} \times x_0^{\text{power}-1}\cdot \text{power}$ 加给 `self.grad` 属性即可，也就是将 `out.grad` 乘以 `power * (self.data ** (power-1))` 加给 `self.grad` 属性。



##### 4. 常用函数
- **ReLU**：
```python
def relu(self):
    out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)

    def _backward():
        if self.requires_grad:
            self.grad = (self.grad or 0) + out.grad * (self.data > 0)

    out._backward = _backward
    out._prev = {self}
    return out
```

反向传播函数根据链式法则计算梯度，不妨设损失函数为 $\mathcal L$ ，对于 ReLU 函数的部分显然有：

$$
\frac{\partial \mathcal L}{\partial x_0} = \frac{\partial \mathcal L}{\partial y} \frac{\partial y}{\partial x_0} = \frac{\partial \mathcal L}{\partial y} \times (x_0 > 0)
$$ 

我们只需要将 $\frac{\partial \mathcal L}{\partial y} \times (x_0 > 0)$ 加给 `self.grad` 属性即可，也就是将 `out.grad` 乘以 `(self.data > 0)` 加给 `self.grad` 属性。

- **均值**：
```python
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
```

反向传播函数根据链式法则计算梯度，不妨设损失函数为 $\mathcal L$ ，对于均值函数的部分显然有：

$$
\frac{\partial \mathcal L}{\partial x_0} = \frac{\partial \mathcal L}{\partial y} \frac{\partial y}{\partial x_0} = \frac{\partial \mathcal L}{\partial y} \times \frac{1}{n}
$$ 

我们只需要将 $\frac{\partial \mathcal L}{\partial y} \times \frac{1}{n}$ 加给 `self.grad` 属性即可，也就是将 `out.grad` 乘以 `1/self.data.size` 加给 `self.grad` 属性。

- **view**：
```pythonview
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
```

反向传播函数根据链式法则计算梯度，不妨设损失函数为 $\mathcal L$ ，则 `view` 函数的梯度就是对原张量的梯度进行一次 `view` 操作，计算如下：

$$
\frac{\partial \mathcal L}{\partial x_0} = \frac{\partial \mathcal L}{\partial y} \frac{\partial y}{\partial x_0} = (\frac{\partial \mathcal L}{\partial y}).view(x_0.shape)
$$ 

我们只需要将 `out.grad` 改变形状后加给 `self.grad` 属性即可。

- **sigmoid**：
```python
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
```

反向传播函数根据链式法则计算梯度，不妨设损失函数为 $\mathcal L$ ，则 sigmoid 函数的梯度就是 sigmoid 函数的导数，计算如下：

$$
\frac{\partial \mathcal L}{\partial x_0} = \frac{\partial \mathcal L}{\partial y} \frac{\partial y}{\partial x_0} = \frac{\partial \mathcal L}{\partial y} \times \sigma(x_0) \times (1 - \sigma(x_0))
$$ 

我们只需要将 $\frac{\partial \mathcal L}{\partial y} \times \sigma(x_0) \times (1 - \sigma(x_0))$ 加给 `self.grad` 属性即可，也就是将 `out.grad` 乘以 `out.data * (1 - out.data)` 加给 `self.grad` 属性。

##### 5. 其他
下面这些操作比较简单，且与张量的基本运算无关，因此不再赘述。

- **字符串表示**：
```python
def __repr__(self):
    return f"Tensor(\ndata = \n{self.data},\nrequires_grad = {self.requires_grad}\n)"
```
- **长度**：
```python
def __len__(self):
    return len(self.data)
```
- **索引**：
```python
def __getitem__(self, idx):
    return Tensor(self.data[idx], requires_grad=self.requires_grad)
```

### 1.3 nn模块

`nn` 模块是 PyTorch 中用于构建神经网络的模块，其主要包含以下内容：

- `Module`：一个抽象类，用于实现神经网络的层。
- `Linear`：一个线性层，用于实现线性变换。
- `Parameter`：一个类，用于实现参数。
- `ReLU`：一个激活函数，用于实现 ReLU 激活函数。
- `MSELoss`：一个损失函数，用于实现均方误差损失函数。
- `Conv2d`：一个卷积层，用于实现二维卷积。
- `CrossEntropyLoss`：一个损失函数，用于实现交叉熵损失函数。
- `AvgPool2d`：一个池化层，用于实现平均池化。
- `BatchNorm2d`：一个批标准化层，用于实现批标准化。
- `sigmoid`：一个激活函数，用于实现 sigmoid 激活函数。
- `SimpleSoftmax`：一个线性层，用于实现简单softmax。

下面我们将分文件介绍 `nn` 模块的实现。

#### 1.3.1 module.py

`Module` 是一个抽象类，用于实现神经网络的层。

##### 类的初始化与基本属性

```python
class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
```

`Module` 类初始化时，会初始化两个字典：`_parameters` 和 `_modules`。`_parameters` 用于保存模型的参数，`_modules` 用于保存子模块。

##### 基本函数重写

- **``__setattr__``**：
```python
    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)
```

`Module` 类重写了 `__setattr__` 方法，用于设置属性。当设置的属性是 `Tensor` 类型时，会将其添加到 `_parameters` 字典中；当设置的属性是 `Module` 类型时，会将其添加到 `_modules` 字典中。


- **``__call__``**：
```python
    def __call__(self, *inputs):
        return self.forward(*inputs)
```

`Module` 类重写了 `__call__` 方法，用于实现模型的前向传播。

##### 其余功能的实现

- **``parameters``**：
```python
    def parameters(self):
        for name, param in self._parameters.items():
            yield param
        for name, module in self._modules.items():
            yield from module.parameters()
```


`parameters` 方法用于返回模型的所有参数。

- **``forward``**：
```python
    def forward(self, *inputs):
        raise NotImplementedError
```

`forward` 方法是一个抽象方法，需要在子类中实现。


- **``children``**：
```python
    def children(self):
        for name, module in self._modules.items():
            yield module
```

`children` 方法用于返回模型的所有子模块。

- **``state_dict``**：
```python
    def state_dict(self):
        state_dict = {}
        for name, module in self._modules.items():
            state_dict[name] = module.state_dict()
        for name, param in self._parameters.items():
            state_dict[name] = param.data
        return state_dict
```

`state_dict` 方法用于返回模型的参数字典。

- **``load_state_dict``**：
```python
    def load_state_dict(self, state_dict):
        for name, obj in state_dict.items():
            if isinstance(obj, numpy.ndarray):
                self._parameters[name].data = obj
                # print(f'Loading parameter {name}')
            else:
                self._modules[name].load_state_dict(obj)
```

`load_state_dict` 方法用于加载模型的参数字典。

- **``save``**：
```python
    def save(self, path):
        state_dict = self.state_dict()
        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)
```


`save` 方法用于保存模型的参数字典。

- **``train``**：
```python
    def train(self):
        for module in self.children():
            if hasattr(module, 'train'):
                module.train()
```

`train` 方法用于设置模型为训练模式。

- **``eval``**：
```python
    def eval(self):
        for module in self.children():
            if hasattr(module, 'eval'):
                module.eval()
```

`eval` 方法用于设置模型为评估模式。

#### 1.3.2 linear.py

`Linear` 是一个线性层，用于实现线性变换。

##### nn.Linear

```python
class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear,self).__init__()
        # self.weight = Parameter(Tensor(np.zeros((in_features, out_features))))
        self.weight = Parameter(Tensor(np.random.randn(in_features, out_features)*0.01))
        self.bias = Parameter(Tensor(np.zeros(out_features)))

    def forward(self, x):
        return x @ self.weight + self.bias
```

`Linear` 类继承自 `Module` 类，实现了线性层的前向传播。

##### nn.Parameter

```python
class Parameter(Tensor):
    def __init__(self, data):
      super().__init__(data, requires_grad = True)
```

`Parameter` 类继承自 `Tensor` 类，实现了参数的初始化。

#### 1.3.3 activation.py

```python
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
```

`ReLU` 类实现了 ReLU 激活函数，`sigmoid` 类实现了 sigmoid 激活函数，`SimpleSoftmax` 类实现了简单softmax。

#### 1.3.4 loss.py

```python
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
```

`MSELoss` 类实现了均方误差损失函数，比较简单，`CrossEntropyLoss` 类实现了交叉熵损失函数，需要计算 softmax 值，因此需要实现 `backward` 方法。

不妨设损失函数为 $\mathcal L$ ，则反向传播的计算公式为：

$$
\frac{\partial \mathcal L}{\partial x_0} = \frac{\partial \mathcal L}{\partial y} \frac{\partial y}{\partial x_0} = \frac{\partial \mathcal L}{\partial y} \times \frac{\partial \text{softmax}(x_0)}{\partial x_0}
$$ 

其中 $\text{softmax}(x_0)$ 是模型输出的 softmax 值，$\frac{\partial \text{softmax}(x_0)}{\partial x_0}$ 是 softmax 函数的导数。


#### 1.3.5 conv2d.py

##### nn.Conv2d

- **初始化**：
```python
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
```

`Conv2d` 类继承自 `Module` 类，实现了二维卷积层的初始化。

- **前向传播**：
```python
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
```

为了计算卷积操作 (conv2d) 的反向传播的梯度，我们需要了解卷积操作的前向传播和反向传播的基本原理。在此过程中，我们将详细讨论如何计算输入和卷积核的梯度。

###### 前向传播

考虑一个输入张量 `X`，卷积核 `W`，以及卷积操作的输出 `Y`。假设输入张量的形状为 `(N, C, H, W)`，卷积核的形状为 `(F, C, K, K)`，输出张量的形状为 `(N, F, H_out, W_out)`。

其中：
- `N`：批量大小
- `C`：输入通道数
- `H`、`W`：输入的高度和宽度
- `F`：卷积核的数量（输出通道数）
- `K`：卷积核的高度和宽度（假设卷积核是正方形）
- `H_out`、`W_out`：输出的高度和宽度，通常由输入大小、卷积核大小、步幅和填充决定

卷积操作的前向传播公式可以表示为：

\[ Y_{n, f, i, j} = \sum_{c=0}^{C-1} \sum_{u=0}^{K-1} \sum_{v=0}^{K-1} X_{n, c, i+u, j+v} \cdot W_{f, c, u, v} \]

###### 反向传播

在反向传播中，我们需要计算损失函数相对于输入 `X` 和卷积核 `W` 的梯度，即 \(\frac{\partial L}{\partial X}\) 和 \(\frac{\partial L}{\partial W}\)。

假设损失函数为 \( L \)，我们有：

\[ \frac{\partial L}{\partial X} \]
\[ \frac{\partial L}{\partial W} \]

**1. 计算 \(\frac{\partial L}{\partial W}\)**

对于卷积核 `W` 的梯度，我们使用链式法则进行计算：

\[ \frac{\partial L}{\partial W_{f, c, u, v}} = \sum_{n=0}^{N-1} \sum_{i=0}^{H_{out}-1} \sum_{j=0}^{W_{out}-1} \frac{\partial L}{\partial Y_{n, f, i, j}} \cdot \frac{\partial Y_{n, f, i, j}}{\partial W_{f, c, u, v}} \]

\[ \frac{\partial Y_{n, f, i, j}}{\partial W_{f, c, u, v}} = X_{n, c, i+u, j+v} \]

因此：

\[ \frac{\partial L}{\partial W_{f, c, u, v}} = \sum_{n=0}^{N-1} \sum_{i=0}^{H_{out}-1} \sum_{j=0}^{W_{out}-1} \frac{\partial L}{\partial Y_{n, f, i, j}} \cdot X_{n, c, i+u, j+v} \]

**2. 计算 \(\frac{\partial L}{\partial X}\)**

对于输入 `X` 的梯度，同样使用链式法则进行计算：

\[ \frac{\partial L}{\partial X_{n, c, i, j}} = \sum_{f=0}^{F-1} \sum_{u=0}^{K-1} \sum_{v=0}^{K-1} \frac{\partial L}{\partial Y_{n, f, i-u, j-v}} \cdot \frac{\partial Y_{n, f, i-u, j-v}}{\partial X_{n, c, i, j}} \]

\[ \frac{\partial Y_{n, f, i-u, j-v}}{\partial X_{n, c, i, j}} = W_{f, c, u, v} \]

因此：

\[ \frac{\partial L}{\partial X_{n, c, i, j}} = \sum_{f=0}^{F-1} \sum_{u=0}^{K-1} \sum_{v=0}^{K-1} \frac{\partial L}{\partial Y_{n, f, i-u, j-v}} \cdot W_{f, c, u, v} \]

注意：在计算过程中，我们需要考虑边界条件，即当 \(i-u\) 或 \(j-v\) 超出范围时， \(\frac{\partial L}{\partial Y_{n, f, i-u, j-v}}\) 应该被视为零。

##### nn.AvgPool2d

- **初始化**：
```python
class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super(AvgPool2d, self).__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
```

`AvgPool2d` 类继承自 `Module` 类，实现了平均池化层的初始化。


- **前向传播**：
```python
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
```


要实现平均池化（AvgPool2d）操作的反向传播，我们首先需要理解平均池化的前向传播过程，然后才能推导出其反向传播过程。

###### 前向传播

平均池化操作的前向传播过程是将输入张量 `X` 的每个局部区域（由池化窗口大小和步幅定义）进行平均，得到输出张量 `Y`。

假设输入张量 `X` 的形状为 `(N, C, H, W)`，输出张量 `Y` 的形状为 `(N, C, H_out, W_out)`，池化窗口大小为 `(K, K)`，步幅为 `S`，则前向传播的公式为：

\[ Y_{n, c, i, j} = \frac{1}{K \times K} \sum_{u=0}^{K-1} \sum_{v=0}^{K-1} X_{n, c, i \times S + u, j \times S + v} \]

###### 反向传播

在反向传播过程中，我们需要计算损失函数 `L` 相对于输入 `X` 的梯度 \(\frac{\partial L}{\partial X}\)。由于平均池化操作的反向传播比较简单，梯度会均匀地分配到池化窗口内的每个元素。

假设损失函数 `L` 相对于输出 `Y` 的梯度为 `dL/dY`，我们有：

\[ \frac{\partial L}{\partial X_{n, c, i, j}} = \frac{1}{K \times K} \sum_{p=0}^{H_{out}-1} \sum_{q=0}^{W_{out}-1} \delta(i \in [p \times S, p \times S + K)) \cdot \delta(j \in [q \times S, q \times S + K)) \cdot \frac{\partial L}{\partial Y_{n, c, p, q}} \]

其中，\(\delta\) 是指示函数，当条件为真时值为 1，否则为 0。

##### nn.BatchNorm2d

- **初始化**：
```python
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
```

`BatchNorm2d` 类继承自 `Module` 类，实现了批量归一化层的初始化。

- **前向传播**：
```python
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
```

Batch Normalization (批归一化) 是一种加速神经网络训练并提高稳定性的方法。Batch Normalization 的前向传播和反向传播计算涉及许多步骤。下面将详细描述 BatchNorm2d 的前向传播和反向传播过程。

###### 前向传播

Batch Normalization 的前向传播包括以下步骤：

1. **计算批量均值和方差**：
   对每个通道计算输入的均值和方差。
   
   \[
   \mu_c = \frac{1}{N \cdot H \cdot W} \sum_{n=1}^N \sum_{i=1}^H \sum_{j=1}^W X_{n,c,i,j}
   \]
   
   \[
   \sigma_c^2 = \frac{1}{N \cdot H \cdot W} \sum_{n=1}^N \sum_{i=1}^H \sum_{j=1}^W (X_{n,c,i,j} - \mu_c)^2
   \]

2. **归一化**：
   使用计算得到的均值和方差对输入进行归一化。
   
   \[
   \hat{X}_{n,c,i,j} = \frac{X_{n,c,i,j} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}
   \]

3. **缩放和平移**：
   通过可学习的参数 γ 和 β 对归一化的结果进行缩放和平移。
   
   \[
   Y_{n,c,i,j} = \gamma_c \hat{X}_{n,c,i,j} + \beta_c
   \]

###### 反向传播

反向传播过程需要计算损失函数相对于输入 \( X \)、均值 \(\mu\)、方差 \(\sigma^2\)、缩放参数 \(\gamma\) 和平移参数 \(\beta\) 的梯度。

1. **计算相对于输出 \(Y\) 的梯度**：
   
   \[
   \frac{\partial L}{\partial Y_{n,c,i,j}}
   \]

2. **计算相对于 \(\gamma\) 和 \(\beta\) 的梯度**：
   
   \[
   \frac{\partial L}{\partial \gamma_c} = \sum_{n=1}^N \sum_{i=1}^H \sum_{j=1}^W \frac{\partial L}{\partial Y_{n,c,i,j}} \cdot \hat{X}_{n,c,i,j}
   \]
   
   \[
   \frac{\partial L}{\partial \beta_c} = \sum_{n=1}^N \sum_{i=1}^H \sum_{j=1}^W \frac{\partial L}{\partial Y_{n,c,i,j}}
   \]

3. **计算相对于 \(\hat{X}\) 的梯度**：
   
   \[
   \frac{\partial L}{\partial \hat{X}_{n,c,i,j}} = \frac{\partial L}{\partial Y_{n,c,i,j}} \cdot \gamma_c
   \]

4. **计算相对于 \(\sigma^2\) 和 \(\mu\) 的梯度**：
   
   \[
   \frac{\partial L}{\partial \sigma_c^2} = \sum_{n=1}^N \sum_{i=1}^H \sum_{j=1}^W \frac{\partial L}{\partial \hat{X}_{n,c,i,j}} \cdot (X_{n,c,i,j} - \mu_c) \cdot -\frac{1}{2} (\sigma_c^2 + \epsilon)^{-3/2}
   \]
   
   \[
   \frac{\partial L}{\partial \mu_c} = \sum_{n=1}^N \sum_{i=1}^H \sum_{j=1}^W \left( \frac{\partial L}{\partial \hat{X}_{n,c,i,j}} \cdot -\frac{1}{\sqrt{\sigma_c^2 + \epsilon}} \right) + \frac{\partial L}{\partial \sigma_c^2} \cdot \frac{-2}{N \cdot H \cdot W} \sum_{n=1}^N \sum_{i=1}^H \sum_{j=1}^W (X_{n,c,i,j} - \mu_c)
   \]

5. **计算相对于输入 \(X\) 的梯度**：
   
   \[
   \frac{\partial L}{\partial X_{n,c,i,j}} = \frac{\partial L}{\partial \hat{X}_{n,c,i,j}} \cdot \frac{1}{\sqrt{\sigma_c^2 + \epsilon}} + \frac{\partial L}{\partial \sigma_c^2} \cdot \frac{2 (X_{n,c,i,j} - \mu_c)}{N \cdot H \cdot W} + \frac{\partial L}{\partial \mu_c} \cdot \frac{1}{N \cdot H \cdot W}
   \]

### 1.4 optim模块

#### optim.SGD

这个 `SGD` 类实现了随机梯度下降（Stochastic Gradient Descent, SGD）优化算法，并加入了动量（momentum）和权重衰减（weight decay）技术。下面详细解释该类的各个部分及其功能。

##### 类初始化
```python
class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0.9, weight_decay=0.0005):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(param.data) for param in self.parameters]
        self.weight_decay = weight_decay
```

###### 参数解释：
- `parameters`：需要优化的模型参数（权重），通常是 `Tensor` 对象的列表。
- `lr`：学习率，控制每次更新的步长大小，默认为 0.01。
- `momentum`：动量因子，用于加速收敛和避免局部最优解，默认为 0.9。
- `weight_decay`：权重衰减因子，防止过拟合，通过在更新时减去参数值的一个比例来实现，默认为 0.0005。

###### 初始化：
- 将 `parameters` 转换为列表并存储。
- 初始化学习率、动量和权重衰减因子。
- 为每个参数初始化动量变量（velocity），初始值为与参数形状相同的零数组。

##### 参数更新 `step` 方法
```python
def step(self):
    for i, param in enumerate(self.parameters):
        if param.requires_grad and param.grad is not None:
            self.velocities[i] = self.momentum * self.velocities[i] + self.lr * param.grad
            param.data -= self.weight_decay * param.data
            param.data -= self.velocities[i]
```

###### 解释：
- 遍历每个参数 `param`，并检查其是否需要梯度更新（`requires_grad` 为 True）且梯度不为 None。
- 计算动量更新公式：
  - 动量的公式为：\[ v_t = \mu v_{t-1} + \eta g_t \]
  - 其中，\( \mu \) 是动量因子，\( \eta \) 是学习率，\( g_t \) 是当前的梯度。
  - 更新动量：`self.velocities[i] = self.momentum * self.velocities[i] + self.lr * param.grad`
- 应用权重衰减：`param.data -= self.weight_decay * param.data`
  - 这一步会将参数的值减去一个与其自身值成比例的数，从而起到正则化的作用。
- 更新参数值：`param.data -= self.velocities[i]`
  - 这一步将动量更新后的值从参数中减去。

##### 梯度清零 `zero_grad` 方法
```python
def zero_grad(self):
    for param in self.parameters:
        if param.requires_grad:
            param.grad = np.zeros_like(param.data)
```

###### 解释：
- 遍历每个参数 `param`，并检查其是否需要梯度更新。
- 将梯度清零：`param.grad = np.zeros_like(param.data)`
  - 这一步非常重要，因为在每次参数更新之后，需要将梯度清零，以便在下一次迭代中计算新的梯度。

### 1.5 加载数据集相关方法
这些方法比较简单，放在mytorch/__init__.py文件中。下面给出代码，不作赘述。

```python
def load(path):
    with open(path, 'rb') as f:
        state_dict = pickle.load(f)
        return state_dict
    
def read_images(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def read_labels(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def load_mnist(data_dir):
    train_images = read_images(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    train_labels = read_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    test_images = read_images(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    test_labels = read_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    
    return (train_images, train_labels), (test_images, test_labels)
```

## 2. 实现MNIST分类任务
## 3. 实验结果与分析