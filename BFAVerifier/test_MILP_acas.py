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
from acas_xu_gurobi_encoding_BFAVerifier import *
import json

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default="mnist")
# parser.add_argument("--arch", default="3blk_10_10_10")
# parser.add_argument("--arch", default="6blk_50_50_50_50_50")
parser.add_argument("--sample_id", type=int, default=0)
parser.add_argument("--rad", type=int, default=2)
parser.add_argument("--qu_bit", type=int, default=8)
parser.add_argument("--flip_bit", type=int, default=1)
parser.add_argument("--outputPath", default="./MILP_OUTPUTS/")
parser.add_argument("--parameters_file", default=None, type=str, help="The parameters to be proved, filtered by DeepPoly*")
parser.add_argument("--hint_file", type=str, default=None, help="The hint file for the MILP encoding. contains the bounds of many variables")
parser.add_argument("--weightPath", type=str, default=None, help="The path to the weight file")
parser.add_argument("--instance_file", type=str, default=None, help="The instance file for the verification, contains input and label")
args = parser.parse_args()

instance_path = args.instance_file

if "CNT1" in instance_path:
    args.flip_bit = 1
if "CNT2" in instance_path:
    args.flip_bit = 2
if "CNT3" in instance_path:
    args.flip_bit = 3
if "CNT4" in instance_path:
    args.flip_bit = 4

if "CNT" not in instance_path:
    assert(0)
    
instance_path = args.instance_file.replace(f"aCNT{args.flip_bit}", "")


model = DeepModel(
    [50,50,50,50,50,50,5],
    last_layer_signed=True,
)

weight_path = args.weightPath.replace(f"aCNT{args.flip_bit}", "")

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.build((None, 1 * 5))
# model.load_weights(weight_path)

# initialize the encoding variables and some necessary information for each neuron

# "DeltaWs": [0.008536118222033883, 0.003398927177969865, 0.0028792483600105826, 
# 0.0030623686125897985, 0.003627643341154564, 0.0033347747457309032, 
# 0.005637749442904014], "bit_all": 8, 
# "bit_all": 8, "weight_path": "benchmark/reluplex/nnet/nnet_1_3", 
# "method": "binarysearch_all", "signed": 1, 
# "input_upper": [0.6799, 0.5, -0.4984, 0.5, 0.5], 
# "input_lower": [0.269, 0.1114, -0.5, 0.2273, 0.0], 
# "input_id": 10, "label": 0}
with open(instance_path, 'r') as f:
    info = json.load(f)
    x_lb = info['input_lower']
    x_ub = info['input_upper']
    Delta_LL = info['DeltaWs']
    original_prediction = info['label']
    original_output = np.array([0, 0, 0, 0, 0])
    original_output[original_prediction] = 114514
    

############### TODO: load "Delta_LL", "W" ###############

# Delta_LL_path = "benchmark/benchmark_QAT_also_quantized_bias/QAT_{}_{}_qu_{}_accuracy.txt".format(args.dataset, args.arch,
#                                                                                          args.qu_bit)
# f_noIP = open(Delta_LL_path)
# Delta_LL = [float(i) for i in f_noIP.readlines()[-1].split('[')[1].split(']')[0].split(',')]
# Delta_LL = [0.008536118222033883, 0.003398927177969865, 0.0028792483600105826, 0.0030623686125897985, 0.003627643341154564, 0.0033347747457309032, 0.005637749442904014]
############### TODO: load "W" ###############
# W is the set of all parameters that cannot solved by verify_DeepPolyR;
# W's format: a dict
#   key:    the weight index [i,j,k] (第i个hidden layer、第j个neuron，前面layer的第k个variable过来的parameter) or bias index [i,j,k] (第i个layer、第j个neuron，k=None)
#   value:  the set of all flipped values of the original parameter
#   这里给了一个的例子，其中flip_bit = Q

key_list = []
if args.parameters_file is not None:
    with open(args.parameters_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            if line.strip() == "ALL PROVED BY DPA":
                print("ALL PROVED BY DPA")
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
            if bin(i).count("1") <= args.flip_bit:
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
    print("No parameters_file given.")
    
print("key_list: ", key_list)
print("W: ", W)


############### TODO: load "all_lb_LL" and "all_ub_LL" ###############
# 这里给了一个例子, 其中所有的lb都设置为-100, ub都设置为100
# 1. 为了保证正确性，这里给到的lb 和 ub 应给一个保守的范围，即，这里的 lb 应该 比 从DeepPolyR获取得到的lb 小一些，这里的 ub 应该 比 从DeepPolyR获取得到的 ub 小一些
#      由于我不知道 具体是多少，所以这里我暂时给了一个相对保守的 bounds，-100 ~ +100；
# 2. 这个保守的范围越小，MILP的求解效率会更高。
#####################################################
all_lb_LL = [[-100 for i in range(l.units)] for l in model.dense_layers]
all_ub_LL = [[100 for i in range(l.units)] for l in model.dense_layers]





############################## !!!!!!!!!!!! 从这里开始实现 Function: verify_MILP ##############################
# 以下是 MILP 的验证算法入口

# 初始化，定义变量等等
milp_encoding = QNNEncoding_MILP(model, W, Delta_LL, args, all_lb_LL, all_ub_LL)

# original_output = model.predict(np.expand_dims(x_test[args.sample_id], 0))[0]
# original_prediction = 0

# original_output = np.array([114514, 0, 0, 0, 0])

# x_lb = [0.269, 0.1114, -0.5, 0.2273, 0.0]
# x_ub = [0.6799, 0.5, -0.4984, 0.5, 0.5]

print("\nThe output of ground-truth is: ", original_prediction)

# 对input region、output属性、QNN进行编码，以及对整个问题进行求解
if_Robust, counterexample, output, BFA_info = check_robustness_gurobi(
    milp_encoding, args, original_prediction, x_lb, x_ub)

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
