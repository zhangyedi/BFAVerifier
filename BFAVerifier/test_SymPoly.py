import argparse
from utils.deep_layers import *
# from gurobipy import GRB
import numpy as np
from utils.DeepPoly_DNN import *
from utils.fault_tolerent import *
import portalocker
import sys
import csv
import time
import datetime
import json

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default="mnist")
parser.add_argument("--arch", default="1blk_100")
parser.add_argument("--sample_id", type=int, default=0)
parser.add_argument("--rad", type=int, default=2)
parser.add_argument("--bit_all", type=int, default=None) 
parser.add_argument("--bit_int", type=int, default=2)
parser.add_argument("--bit_frac", type=int, default=2)
parser.add_argument("--bit_only_signed", type=bool, default=False, help="Only consider signed-bit flip. This will override the method to baseline")
parser.add_argument("--QAT", type=bool, default=False) # Quantization Aware Training
parser.add_argument("--also_qu_bias", type=bool, default=None)
parser.add_argument("--targets_per_layer", type=int, default=None)
parser.add_argument("--method", choices=["baseline","binarysearch"],type=str,default="baseline")
parser.add_argument("--perf", type=str, default="")
parser.add_argument("--description", type=str, default="")
parser.add_argument("--save_test_path", type=str, default="", help = "save the benchmark infomation for GPU DeepPolyR. Includes layerSizes, weights, bias, inputs and label")

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
DeltaWs = None
if args.bit_all != None:
    if args.QAT:
        if args.also_qu_bias:
            weight_path = f"benchmark/benchmark_QAT_also_quantized_bias/QAT_{args.dataset}_{args.arch}_qu_{args.bit_all}.h5"
            info_path = f"benchmark/benchmark_QAT_also_quantized_bias/QAT_{args.dataset}_{args.arch}_qu_{args.bit_all}_accuracy.txt"
        else:
            weight_path = f"benchmark/benchmark_QAT/QAT_{args.dataset}_{args.arch}_qu_{args.bit_all}.h5"
            info_path = f"benchmark/benchmark_QAT/QAT_{args.dataset}_{args.arch}_qu_{args.bit_all}_accuracy.txt"
        with open(info_path, "r") as f:
            # Loss: 0.09997887909412384
            # sparse_categorical_accuracy: 0.9713000059127808
            # training time: 1243.67733836174
            # scaling_factor_ll: [0.0017607045266959532, 0.0008311917870478621, 0.0011096895324274052, 0.001048389246319138, 0.0010453469599296435, 0.0010610049018188]
            # first find the line starts with scaling_factor_ll
            for line in f:
                if line.startswith("scaling_factor_ll"):
                    # then extract the list
                    DeltaWs = list(map(float, line.split(":")[1].strip().strip("[").strip("]").split(",")))
                    print(f"scaling_factor_ll found = {DeltaWs}") 
                    break
    else:    
        weight_path = "benchmark/{}-qnn/{}_{}_Q={}_weight.h5".format(args.dataset, args.dataset, args.arch, args.bit_all)
    if DeltaWs is None:
        print(f"no scaling_factor_ll found, use default value {1.0/(2**args.bit_frac)}")
        DeltaWs = [1.0/(2**args.bit_frac)] * len(arch)
else:
    weight_path = "benchmark/{}/{}_{}_weight.h5".format(args.dataset, args.dataset, args.arch)

model.compile(  # some configurations during training
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.build((None, 28 * 28))  # force weight allocation, print a lot of information

model.load_weights(weight_path)  # input: 0~255

x_input = x_test[args.sample_id]
model_out = model.predict(np.expand_dims(x_test[args.sample_id], 0))[0]
model_predict = np.argmax(model_out)
original_prediction = y_test[args.sample_id]
print("\nModel output is: ", model_out)
print("\nModel prediction is: ", model_predict)

######################## DeepPoly Start #########################


if args.bit_all == None:
    deepPolyNets_DNN = DP_DNN_network(True)
else:
    bit_all = args.bit_all
    bit_int = args.bit_int
    bit_frac = args.bit_frac

    deepPolyNets_QNN = DP_QNN_network(bit_all,DeltaWs,True)
    deepPolyNets_QNN.load_dnn(model)
    
    x_lb, x_ub = np.clip(x_input - args.rad, 0, 255)/255, np.clip(x_input + args.rad, 0, 255)/255
    spec = specification_mnist_linf(x_input, args.rad, model_predict)
    if args.bit_only_signed:
        args.method = "baseline_only_signed"

    if args.method == "baseline" or args.method == "baseline_only_signed":
        print(args.method)
        if args.targets_per_layer:
            print(f"{args.method} targets_per_layer")
            allowSigned = args.bit_only_signed
            algo = BFA_algo_baseline_targets_per_layer(spec,deepPolyNets_QNN, args.targets_per_layer ,allowSigned)
        else:
            print("only_last_layer")
            algo = BFA_algo_baseline(spec,deepPolyNets_QNN,args.bit_only_signed)
    else:
        print("Binary")
        if args.targets_per_layer:
            print("targets_per_layer")
            allowSigned = args.bit_only_signed
            algo = BFA_algo_binarySearch_targets_per_layer(spec,deepPolyNets_QNN, args.targets_per_layer,deepPolyNets_QNN, allowSigned)
        else:
            print("only_last_layer")
            algo = BFA_algo_binarySearch(spec,deepPolyNets_QNN)
    
    if args.save_test_path != "":
        jdata = algo.network.dump()
        jdata["weight_path"] = weight_path
        jdata["method"] = args.method
        jdata["signed"] = args.bit_only_signed
        jdata["input_upper"] = x_ub.tolist()
        jdata["input_lower"] = x_lb.tolist()
        jdata["input_id"] = args.sample_id
        jdata["label"] = int(model_predict)
        with open(args.save_test_path, "w") as f:
            json.dump(jdata, f)
        print(f"Save the benchmark information to {args.save_test_path}...")
        print("exitting")
        exit(0)
    
    experimentStartDate = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    wallClockTimeStart = time.time()
    processTimeStart = time.process_time()
    
    res = algo.operate()

    elapsedWallClockTime = time.time() - wallClockTimeStart
    elapsedProcessTime = time.process_time() - processTimeStart


    PERF = sys.stdout if args.perf == "" else open("results/{}".format(args.perf), "a")
    writer = csv.writer(PERF)
    if PERF != sys.stdout:
        portalocker.lock(PERF, portalocker.LOCK_EX)
        #if the csv file doest have a top row, add it. remember I have locked the file
        if len(open("results/{}".format(args.perf)).read()) == 0:
            writer.writerow(["Model","Input_Id","Method","rad","result","elapsedWallClockTime","DpolyInvokeTimes","elapedProcessTime","experimentStartDate","spec","keyDetail","Description"])
        
            
    if len(res) == 2:
        assert res[0] == True
        
        #Model,Input_Id,Method,rad,result,elapsedWallClockTime,DpolyInvokeTimes,elapedProcessTime,experimentStartDate,spec,keyDetail,Description
        writer.writerow([weight_path, f"{args.sample_id}", args.method, args.rad, "UNSAT", elapsedWallClockTime, res[-1], elapsedProcessTime, experimentStartDate,spec,"_",args.description])
    else:
        assert res[0] == False
        #Model,Input_Id,Method,rad,result,elapsedWallClockTime,DpolyInvokeTimes,elapedProcessTime,experimentStartDate,spec,ketDetail,Description
        #False,,outputRange,self.dpr_time
        _,layerIndex,neuronIndex,weightIndex,position,outputRange,_ = res
        writer.writerow([weight_path, f"{args.sample_id}", args.method, args.rad, f"SAT_{layerIndex}_{neuronIndex}_{weightIndex}_{position}", elapsedWallClockTime, res[-1], elapsedProcessTime, experimentStartDate,spec,str(outputRange),args.description])
    #close file and release lock
    if PERF != sys.stdout:
        portalocker.unlock(PERF)
        PERF.close()
        
    exit(0)

deepPolyNets_DNN.load_dnn(model)

x_lb, x_ub = np.clip(x_input - args.rad, 0, 255)/255, np.clip(x_input + args.rad, 0, 255)/255

input_size = 784

low = np.array(x_lb, dtype=np.float32) * np.ones(input_size, dtype=np.float32)
high = np.array(x_ub, dtype=np.float32) * np.ones(input_size, dtype=np.float32)

deepPolyNets_DNN.property_region = 1

for i in range(deepPolyNets_DNN.layerSizes[0]):
    deepPolyNets_DNN.layers[0].neurons[i].concrete_lower = x_lb[i]
    deepPolyNets_DNN.layers[0].neurons[i].concrete_upper = x_ub[i]
    deepPolyNets_DNN.property_region *= (x_ub[i] - x_lb[i])
    deepPolyNets_DNN.layers[0].neurons[i].concrete_algebra_lower = np.array([x_lb[i]])
    deepPolyNets_DNN.layers[0].neurons[i].concrete_algebra_upper = np.array([x_ub[i]])
    deepPolyNets_DNN.layers[0].neurons[i].algebra_lower = np.array([x_lb[i]])
    deepPolyNets_DNN.layers[0].neurons[i].algebra_upper = np.array([x_ub[i]])

deepPolyNets_DNN.add_weight_range(1,3,32,-1,0)
deepPolyNets_DNN.deeppoly()

for out_index in range(len(deepPolyNets_DNN.layers[-1].neurons)):
    lb = deepPolyNets_DNN.layers[-1].neurons[out_index].concrete_lower_noClip
    ub = deepPolyNets_DNN.layers[-1].neurons[out_index].concrete_upper_noClip
    print("The output bound for the output neuron",out_index,"is: [",lb,',',ub,']')

deepPolyNets_DNN.clear_weight_range()

deepPolyNets_DNN.deeppoly()

print("===========clear weight variable=============")

for out_index in range(len(deepPolyNets_DNN.layers[-1].neurons)):
    lb = deepPolyNets_DNN.layers[-1].neurons[out_index].concrete_lower_noClip
    ub = deepPolyNets_DNN.layers[-1].neurons[out_index].concrete_upper_noClip
    print("The output bound for the output neuron",out_index,"is: [",lb,',',ub,']')

print("===========added back weight variable=============")

deepPolyNets_DNN.add_weight_range(1,3,32,-1,0)

deepPolyNets_DNN.deeppoly()

for out_index in range(len(deepPolyNets_DNN.layers[-1].neurons)):
    lb = deepPolyNets_DNN.layers[-1].neurons[out_index].concrete_lower_noClip
    ub = deepPolyNets_DNN.layers[-1].neurons[out_index].concrete_upper_noClip
    print("The output bound for the output neuron",out_index,"is: [",lb,',',ub,']')
