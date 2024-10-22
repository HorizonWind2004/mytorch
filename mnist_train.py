import numpy as np
import os
from tqdm import tqdm
import mytorch
from mytorch.tensor import Tensor
from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU, SimpleSoftmax, sigmoid
from mytorch.nn.conv import Conv2d, AvgPool2d, BatchNorm2d#, BatchNorm1d
from mytorch.nn.loss import MSELoss, CrossEntropyLoss
from mytorch.nn.module import Module
from mytorch.optim.sgd import SGD
import argparse

from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import unsharp_mask

# 定义图像变换函数
def augment_image(image):
    # plt.imshow(sharpened_image)
    # plt.show()

    # 随机旋转
    angle = np.random.uniform(-10, 10)
    rotated = transform.rotate(image, angle = angle, mode='constant', preserve_range = True, clip=True)
    
    # 随机平移
    translation = np.random.uniform(-3, 3, 2)
    translated = transform.warp(rotated, transform.AffineTransform(translation=translation), mode = "constant")
    
    # 随机缩放
    scale = np.random.uniform(0.9, 1.1)
    scaled = transform.rescale(translated, scale, mode='constant', anti_aliasing = False, preserve_range = True)
    
    # 裁剪或填充到28x28
    if scaled.shape[0] > 28:
        scaled = transform.resize(scaled, (28, 28), mode='constant', anti_aliasing = False, preserve_range = True)
    elif scaled.shape[0] < 28:
        padded = np.zeros((28, 28))
        start = (28 - scaled.shape[0]) // 2
        padded[start:start+scaled.shape[0], start:start+scaled.shape[1]] = scaled
        scaled = padded
    
    # sharpened_image = unsharp_mask(image, radius=1, amount=1)
    # plt.imshow(sharpened_image)
    # plt.show()
    
    # return sharpened_image
    return scaled

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/MNIST/raw')
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=10) # 10 的倍数！
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--model', type=str, default='LeNet')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--loss_type', type=str, default='CrossEntropyLoss')

args = parser.parse_args()

batch_size = args.batch_size
data_dir = args.data_dir
num_epochs = args.epochs
lr = args.lr
momentum = args.momentum
weight_decay = args.weight_decay

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

class FCN(Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = Linear(28 * 28, 128)
        self.relu1 = ReLU()
        self.fc2 = Linear(128, 64)
        self.relu2 = ReLU()
        self.fc3 = Linear(64, 10)

    def forward(self, x):
        x = x.view(x.data.shape[0], 28 * 28)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


input_shape = (1, 28, 28)  
num_classes = 10

(x_train, y_train), (x_test, y_test) = mytorch.load_mnist(data_dir)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 生成增强后的数据集
augmented_images = []
print('Augmenting images...')
for image in tqdm(x_train):
    # print(image.shape)
    augmented_image = augment_image(image)
    augmented_images.append(augmented_image)

x_train = np.concatenate((x_train, np.array(augmented_images)))
y_train = np.concatenate((y_train, y_train))

# n = 101
# image = x_train[n]
# image = augment_image(image)

# plt.imshow(x_train[n])
# plt.show()
# plt.imshow(image)
# plt.show()
# print(y_train[n])

# print(x_train.shape)

# normalize the image.

mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

print(f'mean: {mean}, std: {std}')

# print(x_train.shape)

x_train.resize((120000 // batch_size, batch_size, *input_shape))
x_test.resize((10000, *input_shape))

y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]
y_train.resize((120000 // batch_size, batch_size, num_classes))
y_test.resize((10000, num_classes))


x_train = Tensor(x_train, requires_grad=True)
y_train = Tensor(y_train, requires_grad=True)
x_test = Tensor(x_test, requires_grad=False)
y_test = Tensor(y_test, requires_grad=False)

if args.model == 'LeNet':
    model = LeNet()
    print('Using LeNet!')
elif args.model == 'FCN':
    model = FCN()
    print('Using FCN!')
else:
    raise NotImplementedError('Invalid model type')

if args.pretrained is not None:
    print(f'Loading pretrained model from {args.pretrained}...')
    # print(mytorch.load(args.pretrained))
    model.load_state_dict(mytorch.load(args.pretrained))
else:
    print('Training from scratch...')

if args.loss_type == 'CrossEntropyLoss':
    criterion = CrossEntropyLoss()
elif args.loss_type == 'MSELoss':
    criterion = MSELoss()
else:
    raise NotImplementedError('Invalid loss type')

optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

max_success_rate = 0
for epoch in tqdm(range(num_epochs)):
    # random shuffle
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]
    model.train()
    for batch_id in tqdm(range(len(x_train))):
        x_batch = x_train[batch_id]
        y_batch = y_train[batch_id]
        output = model(x_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_id % 100 == 0: 
            print(f'Epoch {epoch}, Loss: {loss.data}')
            
        # model.save(f'debug_{args.model}_{epoch}.pkl')
    print(f'Epoch {epoch}, Loss: {loss.data}')
    model.eval()
    model_output_test = model(x_test)
    test_loss = criterion(model_output_test, y_test)
    print(f'Test Loss: {test_loss.data}')
    success = 0
    for pred, true in zip(model_output_test.data, y_test.data):
        if np.argmax(pred) == np.argmax(true):
            success += 1
    print(f'Success rate: {success}/{len(y_test)}')
    if success / len(y_test) > max_success_rate:
        max_success_rate = success / len(y_test)
        model.save(f'{args.model}_{epoch}.pkl')

model_output_test = model(x_test)
test_loss = criterion(model_output_test, y_test)
print(f'Test Loss: {test_loss.data}')

print("Predictions vs True values:")
for pred, true in zip(model_output_test.data, y_test.data):
    print(f'Pred: {pred}, True: {true}')
