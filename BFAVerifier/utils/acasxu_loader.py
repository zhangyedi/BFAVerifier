import utils.DeepPoly_DNN as DeepPoly_DNN 
import json
import os
import numpy as np

def quant_weight(weight_fp, bias_fp, BitWidth):
    weight_int = [
        [[] for _ in weight_fp[0]]
    ]
    bias_int = []
    DeltaWs = [0]
    for i in range(1,len(weight_fp)):
        # the max abs of all element in weight_fp[i]
        flattened = np.array(weight_fp[i]).flatten()
        flattened = np.append(flattened, np.array(bias_fp[i-1]).flatten())
        DeltaWs.append(max(np.abs(flattened)) / (2**(BitWidth-1)-1))
    for i in range(1,len(weight_fp)):
        weight_int.append([])
        bias_int.append([])
        for j in range(len(weight_fp[i])):
            weight_int[i].append([])
            bias_int[i-1].append(round(bias_fp[i-1][j] / DeltaWs[i]))
            for k in range(len(weight_fp[i][j])):
                weight_int[i][j].append(round(weight_fp[i][j][k] / DeltaWs[i]))
    # reconstruct_weight
    weight_reconstruct = [[[] for _ in weight_fp[0]]]
    bias_reconstruct = []
    for i in range(1,len(weight_int)):
        weight_reconstruct.append([])
        bias_reconstruct.append([])
        for j in range(len(weight_int[i])):
            bias_reconstruct[i-1].append(bias_int[i-1][j] * DeltaWs[i])
            weight_reconstruct[i].append([])
            for k in range(len(weight_int[i][j])):
                weight_reconstruct[i][j].append(weight_int[i][j][k] * DeltaWs[i])
    return DeltaWs, weight_int, weight_reconstruct, bias_int, bias_reconstruct

def read_weight_bias(nnetfolder: str):
    input_len = 5
    weights = [[[] for _ in range(input_len)]]
    bias = []
    # b1.txt b2.txt ... will appear in folder/bias
    # w1.txt w2.txt ... will appear in folder/weights
    layersize = len(list(os.listdir(os.path.join(nnetfolder, "weights"))))
    for i in range(layersize):
        with open(os.path.join(nnetfolder, "weights", f"w{i+1}.txt")) as f:
            weights.append(eval(f.read()))
        with open(os.path.join(nnetfolder, "bias", f"b{i+1}.txt")) as f:
            bias.append(eval(f.read()))
    return weights, bias
def load_network_from_folder(nnetfolder:str):
    dpoly = DeepPoly_DNN.DP_DNN_network(True)
    weights, bias = read_weight_bias(nnetfolder)
    dpoly.create_network(weights, bias)
    return dpoly

def load_network_from_folder_quant(nnetfolder:str, BitWidth:int):
    weights, bias = read_weight_bias(nnetfolder)
    DeltaWs, weight_int, weight_reconstruct, bias_int, bias_reconstruct = quant_weight(weights, bias, BitWidth)
    dpoly = DeepPoly_DNN.DP_QNN_network(BitWidth, DeltaWs[1:], True)
    dpoly.create_network(weight_reconstruct, bias_reconstruct)
    return dpoly 

def read_parameters(file_path: str):
    with open(file_path, 'r') as f:
        parameters = eval(f.read())
    return parameters

def load_acasxu_network(json_path):
    with open(json_path, 'r') as f:
        network = json.load(f)
        bounds = eval(network['model']['bounds'])
        x_lb = [x[0] for x in bounds]
        x_ub = [x[1] for x in bounds]
        weights = [ [[] for _ in range(len(x_lb))] ]
        bias = []
        for layer in network['model']['layers']:
            type_ = layer['type']
            assert type_ == 'linear'
            weight = read_parameters(layer["weights"])
            b = read_parameters(layer["bias"])
            weights.append(weight)
            bias.append(b)
        # print(weight)
        # print(bias)
        dpoly = DeepPoly_DNN.DP_DNN_network(True)
        dpoly.create_network(weights, bias)
        return dpoly, x_lb, x_ub, network["assert"]
