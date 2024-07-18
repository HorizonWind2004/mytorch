## 深度学习短学期大作业：`mytorch`

### 1. 环境配置：

```
pip install -r requirements.txt
```
### 2. 使用方法：

#### 2.1 下载 MNIST 数据集：

我们利用 `torchvision` 提供的接口下载 MNIST 数据集（下载数据集应该不算用 `torch` 吧）

```shell
python mnist_download.py
```

会把数据集下载到 `./data` 目录下。

#### 2.2 训练模型：

运行脚本 `mnist_train.py` 即可训练模型。脚本提供如下参数：

```shell
python mnist_train.py [--data_dir] [--pretrained] [--batch_size] [--epochs] [--model] [--lr] [--momentum] [--weight_decay] [--loss_type]
```

其中：

- `--data_dir`：数据集所在目录，默认为 `./data`；

- `--pretrained`：预训练模型所在路径，默认为 `None`。

- `--batch_size`：batch 大小，默认为 `10`，需要输入 `10` 的倍数。

- `--epochs`：训练轮数，默认为 `100`。

- `--model`：模型名称，默认为 `LeNet`。也可以选择 `FCN`，内部是一个简单的全连接网络。

- `--lr`：学习率，默认为 `0.01`。

- `--momentum`：动量，默认为 `0.9`。

- `--weight_decay`：权重衰减，默认为 `0.0005`。

- `--loss_type`：损失函数，默认为 `CrossEntropyLoss`。也可以选择 `MSELoss`。

可以简单按照默认值训练：

```shell
python mnist_train.py
```

每一个 `epoch` 结束后，如果正确率大于历史最高正确率，则保存模型在当前目录下，格式为 `.pkl`。

#### 2.3 测试模型：

运行脚本 `mnist_test.py` 即可测试模型。脚本提供如下参数：

```shell
python mnist_test.py [--data_dir] [--pretrained]
```

- `--data_dir`：数据集所在目录，默认为 `./data`；

- `--pretrained`：预训练模型所在路径，默认为 `LeNet.pkl`

用法举例：

```shell
python mnist_test.py --pretrained=LeNet.pkl
```