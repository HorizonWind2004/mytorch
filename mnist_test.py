import mytorch
from mytorch.tensor import Tensor
from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU, SimpleSoftmax, sigmoid
from mytorch.nn.conv import Conv2d, AvgPool2d, BatchNorm2d#, BatchNorm1d
from mytorch.nn.module import Module
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/MNIST/raw')
parser.add_argument('--pretrained', type=str, default='LeNet.pkl')
args = parser.parse_args()

class LeNet(Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = Conv2d(1, 6, kernel_size=5, padding=2)
        self.norm1 = BatchNorm2d(6)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(6, 16, kernel_size=5) 
        self.norm2 = BatchNorm2d(16)
        self.relu2 = ReLU()
        self.fc1 = Linear(16 * 5 * 5, 120)
        # self.norm3 = BatchNorm1d(120)
        self.relu3 = ReLU()
        self.fc2 = Linear(120, 84)
        # self.norm4 = BatchNorm1d(84)
        self.relu4 = ReLU()
        self.fc3 = Linear(84, 10)
        # self.norm5 = BatchNorm1d(10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = AvgPool2d(kernel_size=2)(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = AvgPool2d(kernel_size=2)(x)
        x = x.view(x.data.shape[0], -1)
        x = self.fc1(x)
        # x = self.norm3(x)
        x = self.relu3(x)
        x = self.fc2(x)
        # x = self.norm4(x)
        x = self.relu4(x)
        x = self.fc3(x)
        # x = self.norm5(x)
        return x
    
model = LeNet()
model.load_state_dict(mytorch.load(args.pretrained))

input_shape = (1, 28, 28)  
num_classes = 10

(x_train, y_train), (x_test, y_test) = mytorch.load_mnist(args.data_dir)
x_test = x_test.astype(np.float32)
x_test = x_test / 255.0
x_test = (x_test - 0.12735) / 0.28439
x_test.resize((10000, *input_shape))
y_test = np.eye(num_classes)[y_test]
y_test.resize((10000, num_classes))
x_test = Tensor(x_test, requires_grad=False)
y_test = Tensor(y_test, requires_grad=False)

success = 0
model_output_test = model(x_test)
for pred, true in zip(model_output_test.data, y_test.data):
    if np.argmax(pred) == np.argmax(true):
        success += 1
print(f'Success rate: {success}/{len(y_test)} = {success/len(y_test) * 100.}%')



