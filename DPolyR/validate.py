import utils.DeepPoly_DNN as DeepPoly_DNN
import numpy as np
from copy import deepcopy

def validate_one():
    weight = [
        [[],[]], #input layer
        [[1,1],[1,-1]], 
        [[1,1],[1,-1]],
        [[1,1],[0,1]],
    ]
    bias = [
        [0,0],
        [0,0],
        [0,0],
        [0,0]
    ]
    dpoly = DeepPoly_DNN.DP_DNN_network(True)
    dpoly.create_network(weight,bias)
    dpoly.property_region = 1
    x_lb = [-1,-1]
    x_ub = [1,1]
    for i in range(dpoly.layerSizes[0]):
        dpoly.layers[0].neurons[i].concrete_lower = x_lb[i]
        dpoly.layers[0].neurons[i].concrete_upper = x_ub[i]
        dpoly.property_region *= (x_ub[i] - x_lb[i])
        dpoly.layers[0].neurons[i].concrete_algebra_lower = np.array([x_lb[i]])
        dpoly.layers[0].neurons[i].concrete_algebra_upper = np.array([x_ub[i]])
        dpoly.layers[0].neurons[i].algebra_lower = np.array([x_lb[i]])
        dpoly.layers[0].neurons[i].algebra_upper = np.array([x_ub[i]])
    
    # dpoly.deeppoly()

    # for out_index in range(len(dpoly.layers[-1].neurons)):
    #     lb = dpoly.layers[-1].neurons[out_index].concrete_lower_noClip
    #     ub = dpoly.layers[-1].neurons[out_index].concrete_upper_noClip
    #     print("The output bound for the output neuron",out_index,"is: [",lb,',',ub,']')

    dpoly.add_weight_range(3,0,1,0.5,2)
    dpoly.deeppoly()

    for out_index in range(len(dpoly.layers[-1].neurons)):
        lb = dpoly.layers[-1].neurons[out_index].concrete_lower_noClip
        ub = dpoly.layers[-1].neurons[out_index].concrete_upper_noClip
        print("The output bound for the output neuron",out_index,"is: [",lb,',',ub,']')
    
    for out_index in range(len(dpoly.layers[-1].neurons)):
        lb_al = dpoly.layers[-1].neurons[out_index].concrete_algebra_lower
        ub_al = dpoly.layers[-1].neurons[out_index].concrete_algebra_upper
        print("The algebra output bound for the output neuron",out_index,"is:", lb_al, ub_al)
    
    dpoly.clear_weight_range()
    dpoly.add_weight_range(3,0,0,0,2)

    dpoly.deeppoly()

    for out_index in range(len(dpoly.layers[-1].neurons)):
        lb = dpoly.layers[-1].neurons[out_index].concrete_lower_noClip
        ub = dpoly.layers[-1].neurons[out_index].concrete_upper_noClip
        print("The output bound for the output neuron",out_index,"is: [",lb,',',ub,']')
    
    for out_index in range(len(dpoly.layers[-1].neurons)):
        lb_al = dpoly.layers[-1].neurons[out_index].concrete_algebra_lower
        ub_al = dpoly.layers[-1].neurons[out_index].concrete_algebra_upper
        print("The algebra output bound for the output neuron",out_index,"is:", lb_al, ub_al)


def validate_two():
    weight = [
        [[],[]], #input layer
        [[1,1],[1,-1]], 
        [[1,1],[1,-1]],
        [[1,1],[0,0]],
    ]
    bias = [
        [0,0],
        [0,0],
        [0,0],
        [0,0]
    ]
    dpoly = DeepPoly_DNN.DP_DNN_network(True)
    dpoly.create_network(weight,bias)
    # print("inputSize=",dpoly.inputSize)
    dpoly.property_region = 1
    x_lb = [-1,0]
    x_ub = [3,19]
    for i in range(dpoly.layerSizes[0]):
        dpoly.layers[0].neurons[i].concrete_lower = x_lb[i]
        dpoly.layers[0].neurons[i].concrete_upper = x_ub[i]
        dpoly.property_region *= (x_ub[i] - x_lb[i])
        dpoly.layers[0].neurons[i].concrete_algebra_lower = np.array([x_lb[i]])
        dpoly.layers[0].neurons[i].concrete_algebra_upper = np.array([x_ub[i]])
        dpoly.layers[0].neurons[i].algebra_lower = np.array([x_lb[i]])
        dpoly.layers[0].neurons[i].algebra_upper = np.array([x_ub[i]])
    
    # dpoly.deeppoly()

    # for out_index in range(len(dpoly.layers[-1].neurons)):
    #     lb = dpoly.layers[-1].neurons[out_index].concrete_lower_noClip
    #     ub = dpoly.layers[-1].neurons[out_index].concrete_upper_noClip
    #     print("The output bound for the output neuron",out_index,"is: [",lb,',',ub,']')

    # dpoly.add_weight_range(3,0,1,0.5,1)
    dpoly.deeppoly()

    for out_index in range(len(dpoly.layers[-1].neurons)):
        lb = dpoly.layers[-1].neurons[out_index].concrete_lower_noClip
        ub = dpoly.layers[-1].neurons[out_index].concrete_upper_noClip
        print("The output bound for the output neuron",out_index,"is: [",lb,',',ub,']')
    
    for out_index in range(len(dpoly.layers[-1].neurons)):
        lb_al = dpoly.layers[-1].neurons[out_index].concrete_algebra_lower
        ub_al = dpoly.layers[-1].neurons[out_index].concrete_algebra_upper
        print("The algebra output bound for the output neuron",out_index,"is:", lb_al, ub_al)
    
    dpoly.clear_weight_range()


def validate_network1(lb,ub, affineLayerIndex, neuronIndex, weightIndex, rangeMin, rangeMax):
    weight = [
        [[],[]],
        [[1,1],[1,-1]], 
        [[1,1],[1,-1]],
        [[1,1],[1,-1]],
    ]
    bias = [
        [0,0],
        [0,0],
        [0,0],
        [0,0]
    ]
    dpoly = DeepPoly_DNN.DP_DNN_network(True)
    dpoly.create_network(weight,bias)
    dpoly.add_difference_layer(0)
    dpoly.property_region = 1
    x_lb = deepcopy(lb)
    x_ub = deepcopy(ub)
    for i in range(dpoly.layerSizes[0]):
        dpoly.layers[0].neurons[i].concrete_lower = x_lb[i]
        dpoly.layers[0].neurons[i].concrete_upper = x_ub[i]
        dpoly.property_region *= (x_ub[i] - x_lb[i])
        dpoly.layers[0].neurons[i].concrete_algebra_lower = np.array([x_lb[i]])
        dpoly.layers[0].neurons[i].concrete_algebra_upper = np.array([x_ub[i]])
        dpoly.layers[0].neurons[i].algebra_lower = np.array([x_lb[i]])
        dpoly.layers[0].neurons[i].algebra_upper = np.array([x_ub[i]])

    # dpoly.add_weight_range(affineLayerIndex,neuronIndex,weightIndex,rangeMin,rangeMax)
    dpoly.deeppoly()

    # for out_index in range(len(dpoly.layers[-1].neurons)):
    #     lb = dpoly.layers[-1].neurons[out_index].concrete_lower_noClip
    #     ub = dpoly.layers[-1].neurons[out_index].concrete_upper_noClip
    #     print("The output bound for the output neuron",out_index,"is: [",lb,',',ub,']')
    
    # for out_index in range(len(dpoly.layers[-1].neurons)):
    #     lb_al = dpoly.layers[-1].neurons[out_index].concrete_algebra_lower
    #     ub_al = dpoly.layers[-1].neurons[out_index].concrete_algebra_upper
    #     print("The algebra output bound for the output neuron",out_index,"is:", lb_al, ub_al)
    
    LL = dpoly.layers[-1].neurons[1].concrete_lower_noClip
    BB = dpoly.layers[-1].neurons[1].concrete_upper_noClip
    
    print(f"[{LL},{BB}]")
    
    dpoly.clear_weight_range()

def validate_network2(lb,ub, affineLayerIndex, neuronIndex, rangeMin, rangeMax):
    weight = [
        [[],[]],
        [[1,1],[1,-1]], 
        [[1,1],[1,-1]],
        [[1,1],[1,-1]],
    ]
    bias = [
        [0,0],
        [0,0],
        [0,0],
        [0,0]
    ]
    dpoly = DeepPoly_DNN.DP_DNN_network(True)
    dpoly.create_network(weight,bias)
    dpoly.add_difference_layer(0)
    dpoly.property_region = 1
    x_lb = deepcopy(lb)
    x_ub = deepcopy(ub)
    for i in range(dpoly.layerSizes[0]):
        dpoly.layers[0].neurons[i].concrete_lower = x_lb[i]
        dpoly.layers[0].neurons[i].concrete_upper = x_ub[i]
        dpoly.property_region *= (x_ub[i] - x_lb[i])
        dpoly.layers[0].neurons[i].concrete_algebra_lower = np.array([x_lb[i]])
        dpoly.layers[0].neurons[i].concrete_algebra_upper = np.array([x_ub[i]])
        dpoly.layers[0].neurons[i].algebra_lower = np.array([x_lb[i]])
        dpoly.layers[0].neurons[i].algebra_upper = np.array([x_ub[i]])

    dpoly.add_bias_range(affineLayerIndex,neuronIndex,rangeMin,rangeMax)
    dpoly.deeppoly()

    # for out_index in range(len(dpoly.layers[-1].neurons)):
    #     lb = dpoly.layers[-1].neurons[out_index].concrete_lower_noClip
    #     ub = dpoly.layers[-1].neurons[out_index].concrete_upper_noClip
    #     print("The output bound for the output neuron",out_index,"is: [",lb,',',ub,']')
    
    # for out_index in range(len(dpoly.layers[-1].neurons)):
    #     lb_al = dpoly.layers[-1].neurons[out_index].concrete_algebra_lower
    #     ub_al = dpoly.layers[-1].neurons[out_index].concrete_algebra_upper
    #     print("The algebra output bound for the output neuron",out_index,"is:", lb_al, ub_al)
    
    LL = dpoly.layers[-1].neurons[1].concrete_lower_noClip
    BB = dpoly.layers[-1].neurons[1].concrete_upper_noClip
    
    print(f"[{LL},{BB}]")
    
    dpoly.clear_weight_range()



def validate_network(lb,ub, affineLayerIndex, neuronIndex, weightIndex, rangeMin, rangeMax):
    weight = [
        [[],[]],
        [[1,1],[1,-1]], 
        [[1,1],[1,-1]],
        [[1,1],[1,-1]],
    ]
    bias = [
        [0,0],
        [0,0],
        [0,0],
        [0,0]
    ]
    dpoly = DeepPoly_DNN.DP_DNN_network(True)
    dpoly.create_network(weight,bias)
    dpoly.add_difference_layer(0)
    dpoly.property_region = 1
    x_lb = deepcopy(lb)
    x_ub = deepcopy(ub)
    for i in range(dpoly.layerSizes[0]):
        dpoly.layers[0].neurons[i].concrete_lower = x_lb[i]
        dpoly.layers[0].neurons[i].concrete_upper = x_ub[i]
        dpoly.property_region *= (x_ub[i] - x_lb[i])
        dpoly.layers[0].neurons[i].concrete_algebra_lower = np.array([x_lb[i]])
        dpoly.layers[0].neurons[i].concrete_algebra_upper = np.array([x_ub[i]])
        dpoly.layers[0].neurons[i].algebra_lower = np.array([x_lb[i]])
        dpoly.layers[0].neurons[i].algebra_upper = np.array([x_ub[i]])

    dpoly.add_weight_range(affineLayerIndex,neuronIndex,weightIndex,rangeMin,rangeMax)
    dpoly.deeppoly()

    # for out_index in range(len(dpoly.layers[-1].neurons)):
    #     lb = dpoly.layers[-1].neurons[out_index].concrete_lower_noClip
    #     ub = dpoly.layers[-1].neurons[out_index].concrete_upper_noClip
    #     print("The output bound for the output neuron",out_index,"is: [",lb,',',ub,']')
    
    # for out_index in range(len(dpoly.layers[-1].neurons)):
    #     lb_al = dpoly.layers[-1].neurons[out_index].concrete_algebra_lower
    #     ub_al = dpoly.layers[-1].neurons[out_index].concrete_algebra_upper
    #     print("The algebra output bound for the output neuron",out_index,"is:", lb_al, ub_al)
    
    LL = dpoly.layers[-1].neurons[1].concrete_lower_noClip
    BB = dpoly.layers[-1].neurons[1].concrete_upper_noClip
    
    print(f"[{LL},{BB}]")
    
    dpoly.clear_weight_range()

def decimal3(x):
    """Recursively convert a number to 3 decimal points in the x

    Args:
        x (int|float|list|list[...]|tuple): data
    """
    if isinstance(x, (int, float,np.int64,np.float32,np.float64)):
        if isinstance(x, int):
            return x
        return round(x, 2)
    if x is None:
        return None
    #is tuple
    if isinstance(x, tuple):
        return tuple(decimal3(i) for i in x)
    #is list
    return [decimal3(i) for i in x]

def validate_network_bias(lb,ub, affineLayerIndex, neuronIndex, rangeMin, rangeMax):
    weight = [
        [[],[]],
        [[1,1],[1,-1]], 
        [[1,1],[1,-1]],
        [[1,1],[1,-1]],
    ]
    bias = [
        [0,0],
        [0,0],
        [0,0],
        [0,0]
    ]
    dpoly = DeepPoly_DNN.DP_DNN_network(True)
    dpoly.create_network(weight,bias)
    dpoly.add_difference_layer(0)
    dpoly.property_region = 1
    x_lb = deepcopy(lb)
    x_ub = deepcopy(ub)
    for i in range(dpoly.layerSizes[0]):
        dpoly.layers[0].neurons[i].concrete_lower = x_lb[i]
        dpoly.layers[0].neurons[i].concrete_upper = x_ub[i]
        dpoly.property_region *= (x_ub[i] - x_lb[i])
        dpoly.layers[0].neurons[i].concrete_algebra_lower = np.array([x_lb[i]])
        dpoly.layers[0].neurons[i].concrete_algebra_upper = np.array([x_ub[i]])
        dpoly.layers[0].neurons[i].algebra_lower = np.array([x_lb[i]])
        dpoly.layers[0].neurons[i].algebra_upper = np.array([x_ub[i]])
    
    dpoly.add_bias_range(affineLayerIndex,neuronIndex,rangeMin,rangeMax)
    dpoly.deeppoly()

    # for out_index in range(len(dpoly.layers[-1].neurons)):
    #     lb = dpoly.layers[-1].neurons[out_index].concrete_lower_noClip
    #     ub = dpoly.layers[-1].neurons[out_index].concrete_upper_noClip
    #     print("The output bound for the output neuron",out_index,"is: [",lb,',',ub,']')
    
    # for out_index in range(len(dpoly.layers[-1].neurons)):
    #     lb_al = dpoly.layers[-1].neurons[out_index].concrete_algebra_lower
    #     ub_al = dpoly.layers[-1].neurons[out_index].concrete_algebra_upper
    #     print("The algebra output bound for the output neuron",out_index,"is:", lb_al, ub_al)
    
    LL = dpoly.layers[-1].neurons[1].concrete_lower_noClip
    BB = dpoly.layers[-1].neurons[1].concrete_upper_noClip
    
    print(f"[{LL},{BB}]")
    print("===DPR Verbose===")
    for (id,layer) in enumerate(dpoly.layers):
        print(f"Layer {decimal3(id)}")
        for neuron in layer.neurons:
            print(f"weight: {decimal3(neuron.weight)} bias: {decimal3(neuron.bias)}   \tconcrete_lower: {decimal3(neuron.concrete_lower)} \tconcrete_upper: {decimal3(neuron.concrete_upper)}, \talgebra_lower: {decimal3(neuron.algebra_lower)} \talgebra_upper: {decimal3(neuron.algebra_upper)}, \tconcrete_algebra_lower: {decimal3(neuron.concrete_algebra_lower)} \tconcrete_algebra_upper: {decimal3(neuron.concrete_algebra_upper)}")        

    dpoly.clear_weight_range()

# validate_one()

validate_two()

# std::vector<int> layerIndexes =     {1,3,5,1,3,5,1,3,5,1};
# std::vector<int> neuronIndexes =    {0,0,0,0,0,1,1,1,1,1};
# std::vector<int> weightIndexes =    {0,1,0,1,0,1,0,1,0,1};
# std::vector<double> rangeMins =     {-5,-4,-3,-2,-1,0,1,2,3,4};
# std::vector<double> rangeMaxs =     {-4,-3,-2,-1,0,1,2,3,4,5};

trueLayerIndexes = [1,3,5,1,3,5,1,3,5,1]
affineLayerIndex = [t//2 + 1 for t in trueLayerIndexes]
neuronIndexes = [0,0,0,0,0,1,1,1,1,1]
weightIndexes = [0,1,0,1,0,1,0,1,0,1]
rangeMins = [-5,-4,-3,-2,-1,0,1,2,3,4]
rangeMaxs = [-4,-3,-2,-1,0,1,2,3,4,5]
lower = [-1,4]
upper = [-1,4]

for i in range(10):
    if trueLayerIndexes[i] == 1:
        continue
    validate_network1(lower,upper,affineLayerIndex[i],neuronIndexes[i],weightIndexes[i],rangeMins[i],rangeMaxs[i])
    
# for i in range(10):
#     if trueLayerIndexes[i] == 1:
#         continue
#     validate_network2(lower,upper,affineLayerIndex[i],neuronIndexes[i],rangeMins[i],rangeMaxs[i])
    
    

# for i in range(10):
#     if trueLayerIndexes[i] == 1:
#         continue
#     validate_network_bias(lower,upper,affineLayerIndex[i],neuronIndexes[i],rangeMins[i],rangeMaxs[i])
        