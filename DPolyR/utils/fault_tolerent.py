import utils.DeepPoly_DNN as DeepPoly_DNN
import utils.deep_layers as deep_layers
import numpy as np
from copy import deepcopy
import itertools
import random

class specification(object):
    def precondition_set_network(self,network: DeepPoly_DNN.DP_DNN_network):
        assert isinstance(network, DeepPoly_DNN.DP_DNN_network)

    def postcondition_check_output(self,result):
        pass

class specification_mnist_linf(specification):
    """
    Specification for L_INF norm.
    The Specification only contains 
    """
    def __init__(self, x_input, eps, output):
        """
        x_input: the serilized input of the network. 
        eps: [0,255] set the epsilon for the L_INF norm. 
        output: 0~9 the desired output
        """
        x_lb, x_ub = np.clip(x_input - eps, 0, 255)/255, np.clip(x_input + eps, 0, 255)/255
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.eps = eps
        self.output = int(output)
    def __str__(self):
        return f"specification_mnist_linf_{self.eps}_predict_{self.output}"
    def precondition_set_network(self,network: DeepPoly_DNN.DP_DNN_network):
        super(specification_mnist_linf, self).precondition_set_network(network)
        network.property_region = 1
        for i in range(network.layerSizes[0]):
            network.layers[0].neurons[i].concrete_lower = self.x_lb[i]
            network.layers[0].neurons[i].concrete_upper = self.x_ub[i]
            network.property_region *= (self.x_ub[i] - self.x_lb[i])
            network.layers[0].neurons[i].concrete_algebra_lower = np.array([self.x_lb[i]])
            network.layers[0].neurons[i].concrete_algebra_upper = np.array([self.x_ub[i]])
            network.layers[0].neurons[i].algebra_lower = np.array([self.x_lb[i]])
            network.layers[0].neurons[i].algebra_upper = np.array([self.x_ub[i]])
    
    def postcondition_check_output(self,result:list):
        """
        result: the output ranges of the network
        returns
        """
        assert len(result) == 10
        lb = result[self.output][0]
        ub = [res[1] for res in result]
        # print(result)
        return lb >= max(*ub[:self.output], *ub[self.output+1:])

def getBinaryExpr(int_value, bit) -> int:
    """
    Get the binary representation of the int value. 
    """
    # Check if the int_value is within the valid range
    assert int_value < 2 ** (bit - 1)
    assert int_value >= -2 ** (bit - 1)
    if int_value < 0:
        int_value = (1 << bit) + int_value
    return int(bin(int_value)[2:].zfill(bit),2)


def getIntValue(bin_new_repre,bit_all,):
    return bin_new_repre // (2 ** (bit_all - 1)) * (-2**(bit_all - 1)) + bin_new_repre % (2 ** (bit_all - 1))

def flip_bit(int_value,k,bit_all):
    """
    Flip the k-th bit of the signed int representation (completement)
    """
    int_value = int(int_value)
    binaryExpr = getBinaryExpr(int_value,bit_all)
    mask = 2 ** k
    binaryExpr = binaryExpr ^ mask
    return getIntValue(binaryExpr,bit_all)

def flip_bit_mask(int_value, mask:int, bit_all:int):
    """Flip the int_value with mask. two's complement

    Args:
        int_value (int/float): target flip
        mask (int): mask
        bit_all (int): bit width
    """
    int_value = int(int_value)
    binaryExpr = getBinaryExpr(int_value,bit_all)
    binaryExpr = binaryExpr ^ mask
    return getIntValue(binaryExpr,bit_all)

def rangeFlipKBitIntPreserve(value, bit_all, DeltaW, k):
    """
    Range if at most k bit(s) are flipped in the int. 
    (while preserve the sign bit)
    """
    int_value = value / DeltaW
    assert int_value < 2 ** (bit_all)
    assert int_value > -2 ** (bit_all) - 1
    assert bit_all - 1 >= k
    int_value = int(int_value)
    bin_repre = getBinaryExpr(int_value,bit_all)
    #for mask who has exactly k 1 in the binary repre
    rangeMin = 99999999
    rangeMax = -99999999
    if k == 1:
        rangeMin = int_value
        rangeMax = int_value
        for i in range(bit_all - 1):
            int_new_value = flip_bit(int_value,i,bit_all)
            rangeMin = min(rangeMin,int_new_value)
            rangeMax = max(rangeMax,int_new_value)
    else:
        for mask in range(2**(bit_all-1) + 1):
            if bin(mask).count('1') <= k:
                bin_new_repre = bin_repre ^ mask
                int_new_value = getIntValue(bin_new_repre,bit_all)
                rangeMin = min(rangeMin,int_new_value)
                rangeMax = max(rangeMax,int_new_value)
    return rangeMin * DeltaW,rangeMax * DeltaW

def rangeFlipKBitInt(value, bit_all, DeltaW, k):
    """
    Range if at most k bit(s) are flipped 
    """
    int_value = value * (2 ** bit_all) / DeltaW
    assert int_value < 2 ** (bit_all)
    assert int_value > -2 ** (bit_all) - 1
    assert bit_all >= k
    int_value = int(int_value)
    bin_repre = getBinaryExpr(int_value,bit_all)
    #for mask who has exactly k 1 in the binary repre
    rangeMin = 99999999
    rangeMax = -99999999
    if k == 1:
        for i in range(bit_all):
            int_new_value = flip_bit(int_value,i,bit_all)
            rangeMin = min(rangeMin,int_new_value)
            rangeMax = max(rangeMax,int_new_value)
    else:
        for mask in range(2**bit_all):
            if bin(mask).count('1') == k:
                bin_new_repre = bin_repre ^ mask
                int_new_value = getIntValue(bin_new_repre,bit_all)
                rangeMin = min(rangeMin,int_new_value)
                rangeMax = max(rangeMax,int_new_value)
                # print(bin(mask))
    return rangeMin * DeltaW,rangeMax * DeltaW

def getRandomAttackTargets(thisLayerSize:int, lastLayerSize:int, targets_n:int, bit_width:int, allow_signed:bool):
    """Get random attack targets

    Returns:
        [(weightIndex,bit_position)] * targets
    """
    assert targets_n <= thisLayerSize * lastLayerSize 
    total_targets = thisLayerSize * lastLayerSize
    # numerized_rands = random.sample(range(total_targets),targets)
    numerized_rands = [i for i in range(targets_n)]
    targets = []
    for numerized_rand in numerized_rands:
        weightIndex = numerized_rand % lastLayerSize
        neuronIndex = numerized_rand // lastLayerSize
        targets.append((neuronIndex,weightIndex))
    return targets
    

class BFA_algo_baseline():
    """
    A baseline_algorithm for verifying the bit-flipped neural network 
    Flip only one bit // more bit //TODO
    """
    def __init__(self, spec: specification_mnist_linf, network: DeepPoly_DNN.DP_QNN_network, onlySigned = False, *args, **kwargs):
        self.network : DeepPoly_DNN.DP_QNN_network = deepcopy(network)
        self.spec : specification_mnist_linf = deepcopy(spec)
        self.bit_all = self.network.bit_all
        self.DeltaWs = deepcopy(self.network.DeltaWs)
        self.dpr_time = 0
        self.only_signed = onlySigned
    
    def operate(self):
        """
        Operate the verification algorithm
        """
        #set precondition
        self.spec.precondition_set_network(self.network)
        affineLayerIndex = 0
        for (layerIndex,layer) in enumerate(self.network.layers):
            layer:DeepPoly_DNN.DP_DNN_layer = layer
            if layer.layer_type == layer.AFFINE_LAYER:
                affineLayerIndex = affineLayerIndex + 1
                if layerIndex != len(self.network.layers) - 1:
                    continue
                for (neuronIndex,neuron) in enumerate(layer.neurons):
                    for (weightIndex,single_weight) in enumerate(neuron.weight):
                        if abs(single_weight) < 1e-10:
                            single_weight = 0.0
                        if self.only_signed:
                            theRange = [self.network.bit_all - 1,]
                        else:
                            theRange = range(self.network.bit_all - 1)
                        for position in theRange:
                            temp = deepcopy(self.network)
                            temp.layers[layerIndex].neurons[neuronIndex].weight[weightIndex] = flip_bit(single_weight / layer.DeltaW ,position ,self.network.bit_all) * layer.DeltaW
                            # print(f"flip {position}-th bit")
                            # print(temp.layers[layerIndex].neurons[neuronIndex].weight[weightIndex])
                            # print("single_weight / layer.DeltaW=", single_weight / layer.DeltaW)
                            # print(flip_bit(single_weight / layer.DeltaW ,position ,self.network.bit_all))
                            # print("layer.DeltaW=",layer.DeltaW)
                            temp.deeppoly()
                            self.dpr_time = self.dpr_time + 1
                            outputRange = [(neuron.concrete_lower_noClip,neuron.concrete_upper_noClip) for neuron in temp.layers[-1].neurons]
                            if not self.spec.postcondition_check_output(outputRange):
                                return False,layerIndex,neuronIndex,weightIndex,position,outputRange,self.dpr_time
        return True,self.dpr_time


class BFA_algo_baseline_targets_per_layer():
    """
    A baseline_algorithm for verifying the bit-flipped neural network 
    specify targets per layer
    consider signed bit
    """
    def __init__(self, spec: specification_mnist_linf, network: DeepPoly_DNN.DP_QNN_network, targets_per_layer, allowSigned = False,  *args, **kwargs):
        self.network : DeepPoly_DNN.DP_QNN_network = deepcopy(network)
        self.spec : specification_mnist_linf = deepcopy(spec)
        self.bit_all = self.network.bit_all
        self.DeltaWs = deepcopy(self.network.DeltaWs)
        self.dpr_time = 0
        self.allowSigned = allowSigned
        self.targets_per_layer = targets_per_layer
    
    def operate(self):
        """
        Operate the verification algorithm
        """
        #set precondition
        self.spec.precondition_set_network(self.network)
        affineLayerIndex = 0
        for (layerIndex,layer) in enumerate(self.network.layers):
            layer:DeepPoly_DNN.DP_DNN_layer = layer
            if layer.layer_type == layer.AFFINE_LAYER:
                affineLayerIndex = affineLayerIndex + 1
                targets = getRandomAttackTargets(layer.size,self.network.layers[layerIndex - 1].size,self.targets_per_layer,self.network.bit_all,self.allowSigned)
                for (neuronIndex,weightIndex) in targets:
                    for position in range(self.network.bit_all):
                        single_weight = layer.neurons[neuronIndex].weight[weightIndex]
                        if abs(single_weight) < 1e-10:
                            single_weight = 0.0
                        temp = deepcopy(self.network)
                        temp.layers[layerIndex].neurons[neuronIndex].weight[weightIndex] = flip_bit(single_weight / layer.DeltaW ,position ,self.network.bit_all) * layer.DeltaW
                        
                        print(f"{layerIndex},{neuronIndex},{weightIndex},{position}")
                        print(f"{single_weight} {single_weight / layer.DeltaW} {layer.DeltaW} {temp.layers[layerIndex].neurons[neuronIndex].weight[weightIndex]}")                        
                        
                        temp.deeppoly()
                        self.dpr_time = self.dpr_time + 1
                        
                        
                        outputRange = [(neuron.concrete_lower_noClip,neuron.concrete_upper_noClip) for neuron in temp.layers[-1].neurons]
                        print(outputRange)
                        if not self.spec.postcondition_check_output(outputRange):
                            return False,layerIndex,neuronIndex,weightIndex,position,outputRange,self.dpr_time
        return True,self.dpr_time


class BFA_algo_binarySearch():
    def __init__(self, spec: specification_mnist_linf, network: DeepPoly_DNN.DP_QNN_network, *args, **kwargs):
        self.network : DeepPoly_DNN.DP_QNN_network = deepcopy(network)
        self.spec : specification_mnist_linf = deepcopy(spec)
        self.bit_all = self.network.bit_all
        self.dpr_times = 0

    def operate(self):
        self.spec.precondition_set_network(self.network)
        affineLayerIndex = 0
        for (layerIndex,layer) in enumerate(self.network.layers):
            layer:DeepPoly_DNN.DP_DNN_layer = layer
            if layer.layer_type == layer.AFFINE_LAYER:
                affineLayerIndex = affineLayerIndex + 1
                if layerIndex != len(self.network.layers) - 1:
                    continue
                for (neuronIndex,neuron) in enumerate(layer.neurons):
                    for (weightIndex,single_weight) in enumerate(neuron.weight):
                        if abs(single_weight) < 1e-10:
                            single_weight = 0.0
                        copyWeight = deepcopy(single_weight)
                        rangeMin,rangeMax = rangeFlipKBitIntPreserve(single_weight,self.bit_all,layer.DeltaW,1)
                        temp = deepcopy(self.network)
                        temp.add_weight_range(affineLayerIndex,neuronIndex,weightIndex,rangeMin,rangeMax)
                        temp.deeppoly()
                        self.dpr_times = self.dpr_times + 1
                        # self.network.clear_weight_range()
                        outputRange = [(neuron.concrete_lower_noClip,neuron.concrete_upper_noClip) for neuron in temp.layers[-1].neurons]
                        # print(f"on the place {affineLayerIndex},{neuronIndex},{weightIndex}")
                        # print("pretest",self.spec.postcondition_check_output(outputRange),outputRange)
                        if not self.spec.postcondition_check_output(outputRange):
                            copyWeight = deepcopy(single_weight)
                            for position in range(self.network.bit_all - 1):
                                # print(single_weight,int(single_weight * 2 ** self.bit_frac), flip_bit(single_weight * 2 ** self.bit_frac,position,self.network.bit_all) / 2 ** self.bit_frac)
                                # single_weight = flip_bit(single_weight * 2 ** self.bit_frac,position,self.network.bit_all) / 2 ** self.bit_frac
                                # print(single_weight)
                                neuron.weight[weightIndex] = flip_bit(single_weight / layer.DeltaW,position,self.network.bit_all) * layer.DeltaW
                                self.network.deeppoly()
                                self.dpr_times = self.dpr_times + 1
                                neuron.weight[weightIndex] = copyWeight
                                outputRange = [(neuron.concrete_lower_noClip,neuron.concrete_upper_noClip) for neuron in self.network.layers[-1].neurons]
                                # print("bittest",self.spec.postcondition_check_output(outputRange),outputRange)
                                if not self.spec.postcondition_check_output(outputRange):
                                    return False,layerIndex,neuronIndex,weightIndex,position,outputRange,self.dpr_times
        return True,self.dpr_times

class BFA_algo_binarySearch_targets_per_layer():
    def __init__(self, spec: specification_mnist_linf, network: DeepPoly_DNN.DP_QNN_network, targets_per_layer , allowSigned = False, *args, **kwargs):
        self.network : DeepPoly_DNN.DP_QNN_network = deepcopy(network)
        self.spec : specification_mnist_linf = deepcopy(spec)
        self.bit_all = self.network.bit_all
        self.dpr_times = 0
        self.targets_per_layer = targets_per_layer
        self.allowSigned = allowSigned

    def operate(self):
        self.spec.precondition_set_network(self.network)
        affineLayerIndex = 0
        for (layerIndex,layer) in enumerate(self.network.layers):
            layer:DeepPoly_DNN.DP_DNN_layer = layer
            if layer.layer_type == layer.AFFINE_LAYER:
                affineLayerIndex = affineLayerIndex + 1
                targets = getRandomAttackTargets(layer.size,self.network.layers[layerIndex - 1].size,self.targets_per_layer,self.network.bit_all,self.allowSigned)
                for (neuronIndex,weightIndex) in targets:
                    single_weight = layer.neurons[neuronIndex].weight[weightIndex]
                    if abs(single_weight) < 1e-10:
                        single_weight = 0.0
                    neuron = layer.neurons[neuronIndex]
                    copyWeight = deepcopy(single_weight)
                    rangeMin,rangeMax = rangeFlipKBitIntPreserve(single_weight,self.bit_all,layer.DeltaW,1)
                    temp = deepcopy(self.network)
                    temp.add_weight_range(affineLayerIndex,neuronIndex,weightIndex,rangeMin,rangeMax)
                    temp.deeppoly()
                    self.dpr_times = self.dpr_times + 1

                    outputRange = [(neuron.concrete_lower_noClip,neuron.concrete_upper_noClip) for neuron in temp.layers[-1].neurons]

                    if not self.spec.postcondition_check_output(outputRange):
                        copyWeight = deepcopy(single_weight)
                        for position in range(self.network.bit_all - 1):

                            neuron.weight[weightIndex] = flip_bit(single_weight / layer.DeltaW ,position ,self.network.bit_all) * layer.DeltaW
                            self.network.deeppoly()
                            self.dpr_times = self.dpr_times + 1
                            neuron.weight[weightIndex] = copyWeight
                            outputRange = [(neuron.concrete_lower_noClip,neuron.concrete_upper_noClip) for neuron in self.network.layers[-1].neurons]

                            if not self.spec.postcondition_check_output(outputRange):
                                return False,layerIndex,neuronIndex,weightIndex,position,outputRange,self.dpr_times
                    
                    #test signed bit
                    
                    temp = deepcopy(self.network)
                    temp.layers[layerIndex].neurons[neuronIndex].weight[weightIndex] = flip_bit(single_weight / layer.DeltaW ,self.network.bit_all - 1 ,self.network.bit_all) * layer.DeltaW
                    temp.deeppoly()
                    self.dpr_times = self.dpr_times + 1
                    outputRange = [(neuron.concrete_lower_noClip,neuron.concrete_upper_noClip) for neuron in temp.layers[-1].neurons]
                    if not self.spec.postcondition_check_output(outputRange):
                        return False,layerIndex,neuronIndex,weightIndex,self.network.bit_all - 1,outputRange,self.dpr_times
                    
                    
        return True,self.dpr_times
