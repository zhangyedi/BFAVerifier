import argparse
from utils.deep_layers import *
# from gurobipy import GRB
import numpy as np
from utils.DeepPoly_DNN import *
from utils.fault_tolerent import *
import sys
import csv
import time
import datetime
from gurobi_encoding_BFAVerifier import *

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default="mnist")
parser.add_argument("--arch", default="3blk_10_10_10")
parser.add_argument("--sample_id", type=int, default=0)
parser.add_argument("--rad", type=int, default=2)
parser.add_argument("--qu_bit", type=int, default=8)
parser.add_argument("--flip_bit", type=int, default=1)
parser.add_argument("--outputPath", default="./MILP_OUTPUTS/")
parser.add_argument("--parameters_file", default=None, type=str, help="The parameters to be proved, filtered by DeepPoly*")
parser.add_argument("--hint_file", type=str, default=None, help="The hint file for the MILP encoding. contains the bounds of many variables")

args = parser.parse_args()

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

archMnist = args.arch.split('_')
numBlk = archMnist[0][:-3]
arch = [784]
blkset = list(map(int, archMnist[1:]))
blkset.append(10)
arch += blkset

assert int(numBlk) == len(blkset) - 1

model = DeepModel(
    blkset,
    last_layer_signed=True,
)

# load parameters from benchmark

weight_path = "benchmark/benchmark_QAT_also_quantized_bias/QAT_{}_{}_qu_{}.h5".format(args.dataset, args.arch,
                                                                                      args.qu_bit)

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.build((None, 28 * 28))
model.load_weights(weight_path)

# initialize the encoding variables and some necessary information for each neuron


############### load "Delta_LL", "W" ###############

Delta_LL_path = "benchmark/benchmark_QAT_also_quantized_bias/QAT_{}_{}_qu_{}_accuracy.txt".format(args.dataset, args.arch,
                                                                                         args.qu_bit)
f_noIP = open(Delta_LL_path)
Delta_LL = [float(i) for i in f_noIP.readlines()[-1].split('[')[1].split(']')[0].split(',')]

############### load "W" ###############
# W is the set of all parameters that cannot solved by verify_DeepPolyR;
# W's format: a dict
#   key:    the weight index [i,j,k] (i-th hidden layer, j-th neuron，k-th input variable) or bias index [i,j,k] (i-th layer, j-th neuron, k=None)
#   value:  the set of all flipped values of the original parameter

key_list = []
if args.parameters_file is not None:
    with open(args.parameters_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            l, i, j = line.split(',')
            l, i, j = int(l), int(i), int(j)
            if j == -1:
                key_list.append(tuple([l, i, None]))
            else:
                key_list.append(tuple([l, i, j]))
        masks = []
        W = dict()
        for i in range(1, 2 ** args.qu_bit):
            if i>=bin(i).count("1") <= args.flip_bit:
                masks.append(i)
        for key in key_list:
            if key[2] is None:
                # original_weights = model.layers[layer somewhat].get_weights()
                original_weights = model.layers[key[0]].get_weights()
                theBias = original_weights[1][key[1]]
                value_list = []
                int_value = round(theBias / Delta_LL[key[0]])
                for mask in masks:
                    value_list.append(flip_bit_mask(int_value,mask,args.qu_bit))
            else:
                original_weights = model.layers[key[0]].get_weights()
                theWeight = original_weights[0][key[2]][key[1]]
                value_list = []
                int_value = round(theWeight / Delta_LL[key[0]])
                for mask in masks:
                    value_list.append(flip_bit_mask(int_value,mask,args.qu_bit))
                    
            W[key] = value_list

else:
    print("No parameters_file given. fallback to sanity check")
    l = 3

    for i in range(10):
        key_list.append(tuple([l, i, 0]))
        key_list.append(tuple([l, i, 1]))
        key_list.append(tuple([l, i, 2]))
        key_list.append(tuple([l, i, 3]))
        key_list.append(tuple([l, i, 4]))
        key_list.append(tuple([l, i, 5]))
        key_list.append(tuple([l, i, 6]))
        key_list.append(tuple([l, i, 7]))
        key_list.append(tuple([l, i, 8]))
        key_list.append(tuple([l, i, 9]))

    for i in range(10):
        key_list.append(tuple([l, i, None]))

    allSum = myCombSum(args.qu_bit, args.flip_bit)
    set_sample = [i - 2 ** (args.qu_bit - 1) + 1 for i in range(allSum)]

    value_list = [set_sample for i in key_list]
    W = dict(zip(key_list, value_list))
    
print("key_list: ", key_list)
print("W: ", W)


all_lb_LL = [[-1000 for i in range(l.units)] for l in model.dense_layers]
all_ub_LL = [[1000 for i in range(l.units)] for l in model.dense_layers]

milp_encoding = QNNEncoding_MILP(model, W, Delta_LL, args, all_lb_LL, all_ub_LL)

original_output = model.predict(np.expand_dims(x_test[args.sample_id], 0))[0]
original_prediction = np.argmax(original_output)

print("\nThe output of ground-truth is: ", model.predict(np.expand_dims(x_test[args.sample_id], 0))[0])

if_Robust, counterexample, output, BFA_info = check_robustness_gurobi(
    milp_encoding, x_test[args.sample_id].flatten(), args, original_prediction)

print(milp_encoding._stats)

if if_Robust == True:
    print(
        "\nThe BFA-tolerant robustness property holds w.r.t. all (1,{})-attack vectors for the input sample {:04d}!".format(
            args.flip_bit, args.sample_id))
elif if_Robust == False:

    print(
        "\nThe BFA-tolerant robustness property does not hold w.r.t. all (1,{})-attack vectors for the input sample {:04d}!".format(
            args.flip_bit, args.sample_id))

    # For validation
    validate_BFA(model, Delta_LL, original_output, counterexample, output, BFA_info, verbose=True)
