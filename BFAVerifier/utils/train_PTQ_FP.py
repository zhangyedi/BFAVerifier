import argparse
import tensorflow_datasets as tfds
from deep_layers import * 

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

numBlk = 5
blkset = [100,100,100,100,100]

blkset.append(10)


def real_round(x):
    if x < 0:
        return np.ceil(x - 0.5)
    elif x > 0:
        return np.floor(x + 0.5)
    else:
        return 0

original_paras = []

int_bit = 2


# Load and preprocess the SVHN dataset
def load_svhn():
    # Load dataset
    (ds_train,ds_test), ds_info = tfds.load('svhn_cropped', split=['train', 'test'], as_supervised=True, with_info=True)
    
    print(ds_info)
    
    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32), label
    def faltten(image, label):
        return tf.reshape(image, [-1]), label
    
    ds_train = ds_train.cache()
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.map(
        faltten, num_parallel_calls=tf.data.AUTOTUNE)
    # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(64)
    
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(
        faltten, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(64)
    
    return ds_train, ds_test, ds_info

ds_train, ds_test, ds_info = load_svhn()

for Q in [4,8]:
    model = DeepModel(
        blkset,
        # input_bits=8,
        # quantization_bits=Q,
        last_layer_signed=True,
    )  # definition of model, print a lot of information

    # print("#################################  222  ##############################")

    # weight_path = "benchmark_PTQ/{}_{}_in_8_relu_{}.h5".format(args.dataset, args.arch, args.relu)
    weight_path = "svhn_5blk_100_100_100_100_100.h5"
    model.build(input_shape=(None, 32 * 32 * 3))  # force weight allocation, print a lot of information
    
    model.compile(  # some configurations during training
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    # model.build((None, 32 * 32 * 3))  # force weight allocation, print a lot of information
    model.build(input_shape=(None, 32 * 32 * 3))  # force weight allocation, print a lot of information
    dummy_input = tf.zeros((1, 32 * 32 * 3))
    model(dummy_input)

    # print("#################################  333  ##############################")
    model.summary()

    # print("#################################  444  ##############################")

    model.load_weights(weight_path)

    # Q = args.qu_bit

    # loss, accu = model.evaluate(x_test, y_test)
    # print("The DNN accu is: ", accu)
    
    if len(original_paras) < 1:
        for i in range(len(blkset)):
            original_paras.append(model.layers[i].get_weights())

    F_W = Q - int_bit 
    F_B = Q - int_bit

    loss, accu = model.evaluate(ds_test)
    print("########################################################## before quant, acc is ", accu)
    deltaWs = []
    for i in range(len(blkset)):
        paras = original_paras[i]  # list of two array
        maxAbsW = np.max(np.abs(np.array(paras[0]).reshape(-1)))
        maxAbsB = np.max(np.abs(np.array(paras[1]).reshape(-1)))
        deltaW = max(maxAbsB, maxAbsW) / (2 ** Q)
        quantizer = lambda a: real_round(a / deltaW) * deltaW
        new_weight = []
        deltaWs.append(deltaW)
        for j in range(len(paras[0])):
            weight_j = paras[0][j].tolist()
            weight_j = list(map(quantizer, weight_j))
            new_weight.append(weight_j)

        new_weight = np.asarray(new_weight)

        bias = paras[1].tolist()
        bias = list(map(quantizer, bias))
        bias = np.asarray(bias)
        model.layers[i].set_weights([new_weight, bias])

    loss, accu = model.evaluate(ds_test)
    print("########################################################## The QNN accu for Q = ", Q, " is: ", accu)
    model.save(f"svhn_5blk_100_100_100_100_100_qu={Q}.h5")
    with open(f"svhn_5blk_100_100_100_100_100_qu={Q}.txt", "w") as f:
        f.write(f"loss: {loss}\n")
        f.write(f"sparse_categorical_accuracy: {accu}\n")
        f.write(f"training time: 0.0\n")
        f.write(f"scaling_factor_ll: {deltaWs}\n")
    # Loss: 0.39896026253700256
    # sparse_categorical_accuracy: 0.8863000273704529
    # training time: 394.2271375656128
    # scaling_factor_ll: [0.04619100786024524, 0.016714421010786486, 0.02659847274903328, 0.03277637497071297]
