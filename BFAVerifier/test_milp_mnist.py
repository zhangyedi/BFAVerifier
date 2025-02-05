import argparse
from utils.deep_layers import *
# from gurobipy import GRB
import numpy as np
from utils.DeepPoly_DNN import *
from utils.fault_tolerent import * 
import sys
import csv
import time
import re
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
parser.add_argument("--res_file", type=str, default=None, help="import information from res file, this will override other args")
parser.add_argument("--hint_only_consistency_check", action='store_true', default= False
                    , help="Execute Consistency check only with bounds from hint files only")
parser.add_argument("--activation_type", type=str, default=None, help="The activation type of the network")

args = parser.parse_args()

key_list = []
weight_path = ""
Delta_LL_path = ""
activation_type = None
if args.res_file is not None:
    para = []
    with open(args.res_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            # (Overall) Fail to prove 4 7 0 with all masks. Summary: 0 1 0
            # (Overall) Fail to prove 4 7 (bias) with all masks. Summary: 0 1 0
            if line.startswith("(Overall) Fail to prove"):
                line = line.strip()
                dp = line.split("Fail to prove ")[1].strip().split(" ")
                i = int(dp[0])
                j = int(dp[1])
                k = None if dp[2] == "(bias)" else int(dp[2])
                para.append((i, j, k))
            if line.startswith("info.layerSizes.size() = "):
                # there is a number following 
                blkNum = int(line.split("info.layerSizes.size() = ")[1]) - 2
            if line.startswith("bias.size()'s are 1 "):
                # there is a number following
                num = re.findall(r"\d+", line)
                blkSize = int(num[1])
            if line.startswith("bit_all is "):
                # there is a number following
                bit_all = int(line.split("bit_all is ")[1])
            if line.startswith("bit_flip_cnt: "):
                # there is a number following
                bit_flip_cnt = int(line.split("bit_flip_cnt: ")[1])
            if line.startswith("input file (optional if contained in json file)"):
                # there is a number like ddd.json, use regex
                # for example, input file (optional if contained in json file): mnistinput//115.json
                num = re.findall(r"\d+", line)
                sample_id = int(num[0])
            if line.startswith("activation_type:"):
                if "relu" in line.lower():
                    activation_type = "relu"
                elif "tanh" in line.lower():
                    activation_type = "tanh"
                elif "sigmoid" in line.lower():
                    activation_type = "sigmoid"
                if activation_type is not None:
                    print("found activation type in res file ", activation_type)
            if line.startswith("weights_path"):
                weight_path = line.split("weights_path: ")[1].strip()
                Delta_LL_path = weight_path.replace(".h5", ".txt")

        if activation_type is None:
            print("activation type not found in res file")
            print("default to relu")
            activation_type = "relu"
        args.activation_type = activation_type

        args.arch = f"{blkNum}blk"+f"_{blkSize}"*blkNum
        args.flip_bit = bit_flip_cnt
        args.qu_bit = bit_all
        args.sample_id = sample_id
        # GPU_QAT_.8.0.5blk_100_100_100_100_100.115.CNT1.TAR-1.json.res
        res_file_filename = args.res_file.split("/")[-1]
        args.rad = int(re.findall(r"\d+", res_file_filename)[1])
        # make para unique
        para = list(set(para))
        # lambda x:(x[0],x[1],str(x[2])) < (x[0],x[1],str(x[2]))
        list.sort(para, key=lambda x: (x[0], x[1], str(x[2])))
        # p[0] - 1, p[1], p[2] if p[2] else -1
        para = [(p[0] - 1, p[1], p[2] if p[2] else -1) for p in para]
        key_list = para
        if len(para) == 0:
            print("ALL PROVED BY DPA")
            print("FINISHED BFA_MILP")
            exit(0)
        print("Import Info from Res file and filename finished")
        print(args)

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
print(blkset)

assert int(numBlk) == len(blkset) - 1

model = DeepModel(
    blkset,
    last_layer_signed=True,
    activation=args.activation_type,
)

# load parameters from benchmark

if len(weight_path) == 0:
    weight_path = "./benchmark/benchmark_PTQ/PTQ_{}_{}_qu_{}.h5".format(args.dataset, args.arch,
                                                                                      args.qu_bit)

    Delta_LL_path = "./benchmark/benchmark_PTQ/PTQ_{}_{}_qu_{}.txt".format(args.dataset, args.arch,
                                                                                         args.qu_bit)

print("weight_path", weight_path)

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.build((None, 28 * 28))
model(x_train[:1])
model.load_weights(weight_path)

# initialize the encoding variables and some necessary information for each neuron



f_noIP = open(Delta_LL_path)
Delta_LL = [float(i) for i in f_noIP.readlines()[-1].split('[')[1].split(']')[0].split(',')]

############### TODO: load "W" ###############
# W is the set of all parameters that cannot solved by verify_DeepPolyR;
# W's format: a dict
#   key:    the weight index [i,j,k] 
#   value:  the set of all flipped values of the original parameter


if args.parameters_file is not None or args.res_file is not None:
    if key_list == []:
        with open(args.parameters_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                if line.strip() == "ALL PROVED BY DPA":
                    print("ALL PROVED BY DPA")
                    print("FINISHED BFA_MILP")
                    exit(0)
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

def readFromHint(hint_file, key, default):
    new_default = []
    if hint_file is None:
        return default
    flg = 0
    with open(hint_file, 'r') as f:
        lines = f.readlines()
        lines = [line.lower().strip() for line in lines  
                 if "[Union of All Fault]".lower() in line.lower()]
        # [Union of All Fault] on affine layer 4 [ (-17.831 , -5.84048), (-8.60663 , 1.75856), (-10.001 , -0.662627), (-4.66571 , 0.426872), (4.9393 , 18.0364), (-7.89183 , -0.72264), (-15.6868 , -2.42713), (-1.60798 , 8.62887), (-5.12368 , 2.06148), (-4.287 , 9.43277),  ]
        # all lower cases. 
        for idx, line in enumerate(lines):
            # make line starts at the second [
            line = line[line.find("[", line.find("[") + 1):]
            list_float = eval(line)
            assert len(list_float) == len(default[idx])
            if key == "lb":
                flg = 1
                new_default.append([item[0]  for item in list_float])
            elif key == "ub":
                flg = 1
                new_default.append([item[1]  for item in list_float])
            else:
                raise ValueError("key should be 'lb' or 'ub'")
    if flg:
        print("read from hint file")
        print(new_default)
        return new_default
    return default

############### TODO: load "all_lb_LL" and "all_ub_LL" ###############

#####################################################
all_lb_LL = [[-1000 for i in range(l.units)] for l in model.dense_layers]
all_ub_LL = [[1000 for i in range(l.units)] for l in model.dense_layers]

all_lb_LL_from_hint = readFromHint(args.hint_file, "lb", all_lb_LL[:])
all_ub_LL_from_hint = readFromHint(args.hint_file, "ub", all_ub_LL[:])

if not args.hint_only_consistency_check:
    all_ub_LL = all_lb_LL_from_hint[:]
    all_ub_LL = all_ub_LL_from_hint[:]



milp_encoding = QNNEncoding_MILP(model, W, Delta_LL, args, all_lb_LL, all_ub_LL)

original_output = model.predict(np.expand_dims(x_test[args.sample_id], 0))[0]
original_prediction = np.argmax(original_output)
ground_truth = y_test[args.sample_id]
print(f"ground_truth={ground_truth}, original_prediction={original_prediction}")

x_lb = np.maximum(x_test[args.sample_id] - args.rad, 0) / 255
x_ub = np.minimum(x_test[args.sample_id] + args.rad, 255) / 255


print("\nThe output of ground-truth is: ", model.predict(np.expand_dims(x_test[args.sample_id], 0))[0])

if_Robust, counterexample, output, BFA_info = check_robustness_gurobi(
    milp_encoding, args, original_prediction, x_lb, x_ub)

print(milp_encoding._stats)

def check_violated(milp_encoding, all_lb_LL, all_ub_LL):
    Min = 1e9
    all_values, all_values_pos = milp_encoding.get_all_neuron_values()
    flg = 0
    assert len(all_values) == len(all_lb_LL)
    for i in range(len(all_values)):
        for j in range(len(all_values[i])):
            if all_values[i][j] < all_lb_LL[i][j] or all_values[i][j] > all_ub_LL[i][j]:
                print(f"Neuron {i},{j} violated: {all_values[i][j]}")
                flg = 1
            print(f"Neuron {i},{j} consistent: {all_lb_LL[i][j]} <= {all_values[i][j]} \t <= {all_values_pos[i][j] if i != len(all_values)-1 else None} <= {all_ub_LL[i][j]}")
            lb = all_lb_LL[i][j]
            ub = all_ub_LL[i][j]
            Min = min(Min, ub-lb)
    print(f"Min: {Min}")
    return flg

# if args.hint_only_consistency_check:
try:
    print("Consistency Check:")
    flg = check_violated(milp_encoding, all_lb_LL_from_hint, all_ub_LL_from_hint)
    rep = "VIOLATED" if flg else "PASSED"
    print(f"Violation Check {rep}")
except:
    print("Consistency Check Failed due to exception in check_violated")
    pass

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

print("FINISHED BFA_MILP")


#pPolyR)> python test_DeepPoly.py --bit_all 4 --QAT 1 --arch 3blk_10_10_10  --method baseline --sample_id 5 --targets_per_layer 1 --description randomtargets --bit_
