import numpy as np
import os
import torch, torchvision

from utilities.data import MNIST, CIFAR10, UR5

    
def init_data_loader(dataset, data_path, batch_size, train=True, training_digits=None, remove_background=None):
    if dataset == "mnist":
        if training_digits is not None:
            return MNIST(data_path, batch_size, train=train, condition_on=[training_digits])
        else:
            return MNIST(data_path, batch_size, train=train)

    elif dataset == "cifar10":
        if training_digits is not None:
            return CIFAR10(data_path, batch_size, train=train, condition_on=[training_digits])
        else:
            return CIFAR10(data_path, batch_size, train=train)

    elif dataset == "ur5":
        if training_digits is not None:
            return UR5(data_path, batch_size, train=train, condition_on=[training_digits], remove_bg=remove_background)
        else:
            return UR5(data_path, batch_size, train=train, remove_bg=remove_background)

def make_dirs(*args):
    for dir in args:
        os.makedirs(dir, exist_ok=True)

def write_preprocessor(config):
    preprocessor = config
    print(vars(preprocessor))
    f = open(config.ckpt_path + "/preprocessor.dat", 'w')
    f.write(str(vars(preprocessor)))
    f.close()

