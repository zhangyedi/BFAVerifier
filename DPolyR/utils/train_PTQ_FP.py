import argparse

from utils.trainer_utils import *
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # import tensorflow as tf
# from torchvision import datasets, transforms
import numpy as np
import time
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--arch", default="1blk_64")
parser.add_argument("--relu", type=int, default=0)  # default=0: relu without upper bound
parser.add_argument("--epoch", type=int, default=1)  # default=0: relu without upper bound
parser.add_argument("--qu_bit", type=int, default=6)  # default=0: relu without upper bound
# parser.add_argument("--resize", type=int, default=10)

args = parser.parse_args()

arch = args.arch.split('_')
numBlk = arch[0][:-3]
blkset = list(map(int, arch[1:]))  # without output layer

blkset.append(10)


def real_round(x):
    if x < 0:
        return np.ceil(x - 0.5)
    elif x > 0:
        return np.floor(x + 0.5)
    else:
        return 0


if args.dataset == "fashion-mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
elif args.dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
else:
    raise ValueError("Unknown dataset '{}'".format(args.dataset))

y_train = y_train.flatten()  # change into one-dimension
y_test = y_test.flatten()

x_train = x_train.reshape([-1, 28 * 28]).astype(np.float32)  # (60000,784), before reshape: (60000,28,28)
x_test = x_test.reshape([-1, 28 * 28]).astype(np.float32)

original_paras = []

int_bit = 2


for Q in [4, 6, 8, 10]:
    model = QuantizedModel(
        blkset,
        input_bits=8,
        quantization_bits=Q,
        last_layer_signed=True,
    )  # definition of model, print a lot of information

    # print("#################################  222  ##############################")

    weight_path = "benchmark_PTQ/{}_{}_in_8_relu_{}.h5".format(args.dataset, args.arch, args.relu)

    model.compile(  # some configurations during training
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # print("#################################  333  ##############################")

    model.build((None, 28 * 28))  # force weight allocation, print a lot of information

    # print("#################################  444  ##############################")

    model.load_weights(weight_path)

    # Q = args.qu_bit

    # loss, accu = model.evaluate(x_test, y_test)
    # print("The DNN accu is: ", accu)

    if len(original_paras) < 1:
        for i in range(len(blkset)):
            original_paras.append(model.layers[i].get_weights())

    F_W = Q - int_bit # Q: 全部的bit数量，int_bit是为整数部分保留的bit数量，F_W是为 小数部分保留的bit数量
    F_B = Q - int_bit

    for i in range(len(blkset)):
        paras = original_paras[i]  # list of two array
        new_weight = []
        for j in range(len(paras[0])):
            weight_j = paras[0][j].tolist()
            weight_j = list(map(lambda a: real_round(a * (2 ** F_W)) / (2 ** F_W), weight_j))
            new_weight.append(weight_j)

        new_weight = np.asarray(new_weight)

        bias = paras[1].tolist()
        bias = list(map(lambda a: real_round(a * (2 ** F_B)) / (2 ** F_B), bias))
        bias = np.asarray(bias)
        model.layers[i].set_weights([new_weight, bias])

    loss, accu = model.evaluate(x_test, y_test)
    print("########################################################## The QNN accu for Q = ", Q, " is: ", accu)
