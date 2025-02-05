import utils.DeepPoly_DNN as DeepPoly_DNN
import utils.deep_layers as deep_layers
import numpy as np
from copy import deepcopy
import tensorflow as tf
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

class specification_acasxu(specification):
    def __init__(self, x_lb, x_ub, label):
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.label = label

    def precondition_set_network(self, network: DeepPoly_DNN.DP_DNN_network):
        network.property_region = 1
        for i in range(network.layerSizes[0]):
            network.layers[0].neurons[i].concrete_lower = self.x_lb[i]
            network.layers[0].neurons[i].concrete_upper = self.x_ub[i]
            network.property_region *= (self.x_ub[i] - self.x_lb[i])
            network.layers[0].neurons[i].concrete_algebra_lower = np.array([self.x_lb[i]])
            network.layers[0].neurons[i].concrete_algebra_upper = np.array([self.x_ub[i]])
            network.layers[0].neurons[i].algebra_lower = np.array([self.x_lb[i]])
            network.layers[0].neurons[i].algebra_upper = np.array([self.x_ub[i]])
            
    def postcondition_check_output(self, result):
        lb_label = result[self.label][0]
        for i in range(len(result)):
            if i != self.label:
                if result[i][1] >= lb_label:
                    return False
        return True


def getBinaryExpr(int_value, bit) -> int:
    """
    Get the binary representation of the int value. 
    """
    # Check if the int_value is within the valid range
    assert int_value < 2 ** (bit - 1)
    assert int_value >= -2 ** (bit - 1), f"{int_value} not in range {-2 ** (bit - 1)} ~ {2 ** (bit - 1) - 1}"
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
    int_value = round(value / DeltaW)
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
            if bin(mask).count('1') <= k and (mask >> (bit_all - 1)) == 0:
                bin_new_repre = bin_repre ^ mask
                int_new_value = getIntValue(bin_new_repre,bit_all)
                rangeMin = min(rangeMin,int_new_value)
                rangeMax = max(rangeMax,int_new_value)
    return rangeMin * DeltaW,rangeMax * DeltaW

def rangeFlipKBitInt(value, bit_all, DeltaW, k):
    """
    Range if at most k bit(s) are flipped 
    """
    int_value = round(value / DeltaW, 0)
    # print("Intvalue = ", int_value)
    # print("Original", value)
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
            if bin(mask).count('1') <= k:
                bin_new_repre = bin_repre ^ mask
                int_new_value = getIntValue(bin_new_repre,bit_all)
                rangeMin = min(rangeMin,int_new_value)
                rangeMax = max(rangeMax,int_new_value)
                # print(bin(mask))
    # contain original
    rangeMin = min(rangeMin,int_value)
    rangeMax = max(rangeMax,int_value)
    return rangeMin * DeltaW,rangeMax * DeltaW

def rangeFlipKBitIntSignMustFlip(value, bit_all, DeltaW, k):
    """
    Range if at most k bit(s) are flipped in the int. The sign must be flipped.
    """
    int_value = round(value / DeltaW, 0)
    assert int_value < 2 ** (bit_all)
    assert int_value > -2 ** (bit_all) - 1
    assert bit_all >= k
    int_value = int(int_value)
    bin_repre = getBinaryExpr(int_value,bit_all)
    # flip the sign
    val = flip_bit(int_value, bit_all - 1, bit_all) * DeltaW
    print(value, val, bit_all, int_value, bin_repre)
    if k == 1:
        return (val,val)
    else:
        return rangeFlipKBitIntPreserve(val, bit_all, DeltaW, k - 1)

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


class BFA_algo_All():
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

def union_bounds(bounds_lhs, bounds_rhs):
    """union bounds. bounds_lhs is 2D array whose entry is (lower,upper)
    """
    if bounds_lhs is None:
        return bounds_rhs[:]
    if bounds_rhs is None:
        return bounds_lhs[:]
    assert len(bounds_lhs) == len(bounds_rhs), f"bounds_lhs {len(bounds_lhs)} != bounds_rhs {len(bounds_rhs)}"
    def union_one_layer(lhs, rhs):
        assert len(lhs) == len(rhs), f"lhs {len(lhs)} != rhs {len(rhs)}"
        return [(min(*lhs_item, *rhs_item), max(*lhs_item, *rhs_item)) for lhs_item,rhs_item in zip(lhs, rhs)]
    res = []
    for lhs, rhs in zip(bounds_lhs, bounds_rhs):
        res.append(union_one_layer(lhs, rhs))
    return res        

weighted_input_relax = DeepPoly_DNN.weighted_input_relax
sigmoid_abstract = DeepPoly_DNN.sigmoid_abstract

def setInput(dpoly:DeepPoly_DNN.DP_QNN_network, x_lb, x_ub):
    dpoly.property_region = 1
    for i in range(dpoly.layerSizes[0]):
        dpoly.layers[0].neurons[i].concrete_lower = x_lb[i]
        dpoly.layers[0].neurons[i].concrete_upper = x_ub[i]
        dpoly.property_region *= (x_ub[i] - x_lb[i])
        dpoly.layers[0].neurons[i].concrete_algebra_lower = np.array([x_lb[i]])
        dpoly.layers[0].neurons[i].concrete_algebra_upper = np.array([x_ub[i]])
        dpoly.layers[0].neurons[i].algebra_lower = np.array([x_lb[i]])
        dpoly.layers[0].neurons[i].algebra_upper = np.array([x_ub[i]])

def test_deeppoly(deeppolyNet:DeepPoly_DNN.DP_QNN_network, x_lb, x_ub, label):
    dpoly = deepcopy(deeppolyNet)
    dpoly.add_difference_layer(label)
    
    setInput(dpoly,x_lb,x_ub)
    dpoly.deeppoly()
    
    LL = dpoly.layers[-1].neurons[1].concrete_lower_noClip
    BB = dpoly.layers[-1].neurons[1].concrete_upper_noClip
    
    # print(f"[{LL},{BB}]")
    lbs_original = [round(neuron.concrete_lower_noClip,5) for neuron in dpoly.layers[-2].neurons]
    ubs_original = [round(neuron.concrete_upper_noClip,5) for neuron in dpoly.layers[-2].neurons]
    LB_original = zip(lbs_original,ubs_original)
    print("DeepPoly (CPU) result is:")
    print(*LB_original)
    print("DeepPoly (CPU) differential result is:")
    lbs = [round(neuron.concrete_lower_noClip,5) for idx,neuron in enumerate(dpoly.layers[-1].neurons) if idx != label]
    ubs = [round(neuron.concrete_upper_noClip,5) for idx,neuron in enumerate(dpoly.layers[-1].neurons) if idx != label]
    LB = zip(lbs,ubs)
    print(*LB)
    return lbs_original, ubs_original

def test_sympoly(deeppolyNet:DeepPoly_DNN.DP_QNN_network, x_lb, x_ub, label, bit_all, bit_flip_cnt, save_union_intervals):
    dpoly = deepcopy(deeppolyNet)
    dpoly.add_difference_layer(label)
    
    setInput(dpoly,x_lb,x_ub)
    
    neurons_to_test = [0,1,2,3,4]
    weights_to_test = [0,1,2,3,4]
    
    union_intervals = None
    def record_intervals(temp:DeepPoly_DNN.DP_DNN_network):
        if save_union_intervals:
            nonlocal union_intervals
            union_intervals = union_bounds(union_intervals, temp.get_intermediate_bounds())
    
    def print_intervals(temp):
        intermediate_bounds = temp.get_intermediate_bounds()
        post_activation_bounds = temp.get_post_activation_bounds()
        for idx,layer_intervals in enumerate(intermediate_bounds):
            print(f"[integrate test] [intermediate bounds] on affine layer {idx+1} {layer_intervals}")
            # if idx != len(intermediate_bounds) - 1:
            #     print(f"[integrate test] [intermediate bounds] on affine layer {idx+1} {post_activation_bounds[idx]}")
    
    if save_union_intervals:
        temp = deepcopy(dpoly)
        temp.deeppoly()
        record_intervals(temp)
        print(f"[integrate test] SymPoly (CPU) print before all flipping cases")
        print_intervals(temp)
        outputRange = [(round(neuron.concrete_lower_noClip,4),round(neuron.concrete_upper_noClip,4)) \
                        for idx,neuron in enumerate(temp.layers[-1].neurons)
                            if idx != label ]
        print(f"lowerbound is ",*outputRange)
        del temp

    
    affineLayerIndex = 0
    for (layerIndex,layer) in enumerate(dpoly.layers):
        layer:DeepPoly_DNN.DP_DNN_layer = layer
        if layer.layer_type == layer.AFFINE_LAYER:
            affineLayerIndex = affineLayerIndex + 1
            if layer.auxLayer:
                continue
            for (neuronIndex,neuron) in enumerate(layer.neurons):
                if neuronIndex not in neurons_to_test:
                    continue
                for (weightIndex,single_weight) in enumerate(neuron.weight):
                    if weightIndex not in weights_to_test:
                        continue
                    # #! debug 2 0 0 
                    # if not(affineLayerIndex == 2 and neuronIndex == 0 and weightIndex == 0):
                    #     continue
                    if abs(single_weight) < 1e-10:
                        single_weight = 0.0
                    print(f"[integrate test] SymPoly (CPU) flipping on {affineLayerIndex} {neuronIndex} {weightIndex}, " +
                          f"Flt={round(single_weight,4)} Int={round(single_weight / layer.DeltaW,0)}")
                    def sympoly_weight_once(rangeMin, rangeMax):
                        temp = deepcopy(dpoly)
                        temp.add_weight_range(affineLayerIndex,neuronIndex,weightIndex,rangeMin,rangeMax)
                        temp.deeppoly()
                        record_intervals(temp)
                        outputRange = [(round(neuron.concrete_lower_noClip,4),round(neuron.concrete_upper_noClip,4)) 
                                       for idx,neuron in enumerate(temp.layers[-1].neurons) if idx != label]
                        print(f"[integrate test] SymWeight=[{round(rangeMin,4), round(rangeMax,4)}]\t lowerbound is ",*outputRange)
                        print_intervals(temp)
                    
                    rangeMin_1,rangeMax_1 = rangeFlipKBitIntSignMustFlip(single_weight, bit_all, layer.DeltaW, bit_flip_cnt)
                    rangeMin_2,rangeMax_2 = rangeFlipKBitIntPreserve(single_weight, bit_all, layer.DeltaW, bit_flip_cnt)
                    rangeMin_3,rangeMax_3 = rangeFlipKBitInt(single_weight, bit_all, layer.DeltaW, bit_flip_cnt)

                    if rangeMin_1 >= 0:
                        #swap, let neg first to output
                        rangeMin_1,rangeMax_1,rangeMin_2,rangeMax_2 = rangeMin_2,rangeMax_2,rangeMin_1,rangeMax_1

                    sympoly_weight_once(rangeMin_1,rangeMax_1)
                    sympoly_weight_once(rangeMin_2,rangeMax_2)
                    # sympoly_weight_once(rangeMin_3,rangeMax_3)
                    
                # #! debug
                # continue
                # test bias
                print(f"[integrate test] SymPoly (CPU) flipping on {affineLayerIndex} {neuronIndex} bias, " +
                        f"Flt={round(neuron.bias,4)} Int={round(neuron.bias / layer.DeltaW,0)}")

                def sympoly_bias_once(rangeMin, rangeMax):
                        temp = deepcopy(dpoly)
                        temp.add_bias_range(affineLayerIndex,neuronIndex,rangeMin,rangeMax)
                        temp.deeppoly()
                        record_intervals(temp)
                        outputRange = [(round(neuron.concrete_lower_noClip,4),round(neuron.concrete_upper_noClip,4)) 
                                        for idx,neuron in enumerate(temp.layers[-1].neurons) if idx != label]
                        print(f"[integrate test] SymWeight=[{round(rangeMin,4), round(rangeMax,4)}]\t lowerbound is ",*outputRange)
                        print_intervals(temp)
                
                rangeMin_1,rangeMax_1 = rangeFlipKBitIntSignMustFlip(neuron.bias, bit_all, layer.DeltaW, bit_flip_cnt)
                rangeMin_2,rangeMax_2 = rangeFlipKBitIntPreserve(neuron.bias, bit_all, layer.DeltaW, bit_flip_cnt)
                rangeMin_3,rangeMax_3 = rangeFlipKBitInt(neuron.bias, bit_all, layer.DeltaW, bit_flip_cnt)                
                
                if rangeMin_1 >= 0:
                    #swap, let neg first to output
                    rangeMin_1,rangeMax_1,rangeMin_2,rangeMax_2 = rangeMin_2,rangeMax_2,rangeMin_1,rangeMax_1

                sympoly_bias_once(rangeMin_1,rangeMax_1)
                sympoly_bias_once(rangeMin_2,rangeMax_2)
                # sympoly_bias_once(rangeMin_3,rangeMax_3)

    if save_union_intervals:
        print(f"[integrate test] SymPoly (CPU) print after all flipping cases")
        for idx,layer_intervals in enumerate(union_intervals):
            print(f"[integrate test] on affine layer {idx+1} {layer_intervals}")


def test_random_soundness(deepPolyNets_QNN: DeepPoly_DNN.DP_QNN_network, model, x_lb, x_ub, model_predict, args):
        lbs, ubs = test_deeppoly(deepPolyNets_QNN, x_lb, x_ub, model_predict)
        lbs = np.array(lbs)
        ubs = np.array(ubs)

        # randomly generate 10000 samples within the x_lb and x_ub using tensor
        sample_amount = 1000

        # print(model.dense_layers[0].kernel.shape) 
        # # make it zero
        # model.dense_layers[0].kernel.assign(tf.zeros(model.dense_layers[0].kernel.shape))

        x_random_inputs = tf.random.uniform((sample_amount, 784), minval=(x_lb,), maxval=(x_ub,)) * 255

        def random_test(lbs, ubs, model):
            y_random_outputs = model.predict(x_random_inputs)
            # assert np.all(x_lb <= x_random_inputs) and np.all(x_random_inputs <= x_ub)
            # test whether the output of the model is within lbs and ubs
            flg_sound = 1
            print(y_random_outputs.shape)
            if args.rad == 0:
                fp_tol = 1e-4
            else:
                fp_tol = 0
            for i in range(sample_amount):
                if not np.all(np.logical_and(y_random_outputs[i] >= lbs - fp_tol, y_random_outputs[i] <= ubs + fp_tol)):
                    print(f"Random soundness test failed at {i}th sample")
                    print(f"with mask {(y_random_outputs[i] >= lbs, y_random_outputs[i] <= ubs)}")
                    print(f"y_random_outputs[i]={y_random_outputs[i]}")
                    print(f"lbs={lbs}")
                    print(f"ubs={ubs}")
                    flg_sound = 0
                    break
            if flg_sound:
                print("Random soundness test PASSED")
                return True
            else:
                print("Random soundness test FAILED")
                return False

        def change_weight_and_random_test(affineLayerIndex, neuronIndex, weightIndex, rangeMin, rangeMax):
            temp = deepcopy(deepPolyNets_QNN)
            setInput(temp, x_lb, x_ub)
            temp.add_weight_range(affineLayerIndex,neuronIndex,weightIndex,rangeMin,rangeMax)
            temp.deeppoly()
            # return raw lb and ub
            lbs = [neuron.concrete_lower_noClip for idx,neuron in enumerate(temp.layers[-1].neurons)]
            ubs = [neuron.concrete_upper_noClip for idx,neuron in enumerate(temp.layers[-1].neurons)]
            lbs = np.array(lbs)
            ubs = np.array(ubs)

            changed_weight_sample_amount = 10
            sample_points = list(np.linspace(rangeMin, rangeMax, changed_weight_sample_amount, endpoint=True))

            # randomly change the weight of the model at the specified position
            for _idx, val in enumerate(sample_points):
                copy_model_weight = model.dense_layers[affineLayerIndex - 1].kernel.numpy().copy()
                changed_model_weight = copy_model_weight.copy()
                changed_model_weight[weightIndex, neuronIndex] = val
                model.dense_layers[affineLayerIndex - 1].kernel.assign(changed_model_weight)
                res = random_test(lbs=lbs, ubs=ubs, model=model)
                if not res:
                    print(f"Random soundness test failed at {affineLayerIndex} {neuronIndex} {weightIndex} [{_idx}]={val}")
                    return False
                model.dense_layers[affineLayerIndex - 1].kernel.assign(copy_model_weight)

            return True

        def change_bias_and_random_test(affineLayerIndex, neuronIndex, rangeMin, rangeMax):
            temp = deepcopy(deepPolyNets_QNN)
            setInput(temp, x_lb, x_ub)
            temp.add_bias_range(affineLayerIndex,neuronIndex,rangeMin,rangeMax)
            temp.deeppoly()
            # return raw lb and ub
            lbs = [neuron.concrete_lower_noClip for idx,neuron in enumerate(temp.layers[-1].neurons)]
            ubs = [neuron.concrete_upper_noClip for idx,neuron in enumerate(temp.layers[-1].neurons)]
            lbs = np.array(lbs)
            ubs = np.array(ubs)

            changed_weight_sample_amount = 10
            sample_points = list(np.linspace(rangeMin, rangeMax, changed_weight_sample_amount, endpoint=True))

            # randomly change the weight of the model at the specified position
            for _idx, val in enumerate(sample_points):
                copy_model_bias = model.dense_layers[affineLayerIndex - 1].bias.numpy().copy()
                changed_model_bias = copy_model_bias.copy()
                changed_model_bias[neuronIndex] = val
                model.dense_layers[affineLayerIndex - 1].bias.assign(changed_model_bias)
                res = random_test(lbs=lbs, ubs=ubs, model=model)
                if not res:
                    print(f"Random soundness test failed at {affineLayerIndex} {neuronIndex} bias {val}")
                    return False
                model.dense_layers[affineLayerIndex - 1].bias.assign(copy_model_bias)

            return True

        print("First DeepPoly Test")
        res = random_test(lbs=lbs, ubs=ubs, model=model)
        if not res:
            print("First DeepPoly Test failed, exitting")
            exit(1)

        print("SymPoly tests on weight")

        amount_random_inputs = 100
        neurons_to_test = [0,1,2,3,4]
        weights_to_test = [0,1,2,3,4]
        bit_all = args.bit_all
        bit_flip_cnt = args.bit_flip_cnt

        affineLayerIndex = 0
        for (layerIndex,layer) in enumerate(deepPolyNets_QNN.layers):
            layer:DeepPoly_DNN.DP_DNN_layer = layer
            if layer.layer_type == layer.AFFINE_LAYER:
                affineLayerIndex = affineLayerIndex + 1
                if layer.auxLayer:
                    continue
                for (neuronIndex,neuron) in enumerate(layer.neurons):
                    if neuronIndex not in neurons_to_test:
                        continue
                    for (weightIndex,single_weight) in enumerate(neuron.weight):
                        if weightIndex not in weights_to_test:
                            continue
                        if abs(single_weight) < 1e-10:
                            single_weight = 0.0


                        print(f"[soundness test] SymPoly (CPU) flipping on {affineLayerIndex} {neuronIndex} {weightIndex}, " +
                            f"Flt={round(single_weight,4)} Int={round(single_weight / layer.DeltaW,0)}")
                            
                        rangeMin_1,rangeMax_1 = rangeFlipKBitIntSignMustFlip(single_weight, bit_all, layer.DeltaW, bit_flip_cnt)
                        rangeMin_2,rangeMax_2 = rangeFlipKBitIntPreserve(single_weight, bit_all, layer.DeltaW, bit_flip_cnt)

                        if rangeMin_1 >= 0:
                            #swap, let neg first to output
                            rangeMin_1,rangeMax_1,rangeMin_2,rangeMax_2 = rangeMin_2,rangeMax_2,rangeMin_1,rangeMax_1

                        res1 = change_weight_and_random_test(affineLayerIndex, neuronIndex, weightIndex, rangeMin_1, rangeMax_1)
                        res2 = change_weight_and_random_test(affineLayerIndex, neuronIndex, weightIndex, rangeMin_2, rangeMax_2)
                        if not res1 or not res2:
                            print(f"SymPoly tests on weight failed at {affineLayerIndex} {neuronIndex} {weightIndex}")
                            return False


                    # test bias
                    print(f"[soundness test] SymPoly (CPU) flipping on {affineLayerIndex} {neuronIndex} bias, " +
                            f"Flt={round(neuron.bias,4)} Int={round(neuron.bias / layer.DeltaW,0)}")

                    rangeMin_1,rangeMax_1 = rangeFlipKBitIntSignMustFlip(neuron.bias, bit_all, layer.DeltaW, bit_flip_cnt)
                    rangeMin_2,rangeMax_2 = rangeFlipKBitIntPreserve(neuron.bias, bit_all, layer.DeltaW, bit_flip_cnt)

                    if rangeMin_1 >= 0:
                        #swap, let neg first to output
                        rangeMin_1,rangeMax_1,rangeMin_2,rangeMax_2 = rangeMin_2,rangeMax_2,rangeMin_1,rangeMax_1

                    res1 = change_bias_and_random_test(affineLayerIndex, neuronIndex, rangeMin_1, rangeMax_1)
                    res2 = change_bias_and_random_test(affineLayerIndex, neuronIndex, rangeMin_2, rangeMax_2)
                    
                    if not res1 or not res2:
                        print(f"SymPoly tests on bias failed at {affineLayerIndex} {neuronIndex}")
                        return False
        
        print("Passed all random soundness tests")
            


# python test_DeepPoly.py --sample_id 432 --bit_all 8 --QAT 1 --arch 3blk_100_100_100 --method binarysearch --also_qu_bias 1 --rad 1 --save_test_path ~/mnist_weight/1.json