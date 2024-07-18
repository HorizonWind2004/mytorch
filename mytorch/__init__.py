import mytorch.nn as nn
import mytorch.optim as optim
from .tensor import Tensor
import pickle
import os
import numpy as np

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

__all__ = ['nn', 'optim', 'Tensor']