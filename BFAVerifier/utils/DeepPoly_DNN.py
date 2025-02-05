import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from copy import deepcopy
import torch

def weighted_input_relax(wl, wu, xl, xu):
    """Compute the two-line abstract that bounds 
            k1 * x + b1 <= wx <= k2 * x + b2 
    where w and x are intervals that might cross the origin
    """
    assert wl <= wu and xl <= xu, "wl <= wu and xl <= xu"
    if xl == xu:
        if xl >= 0:
            return (wl, 0), (wu, 0)
        elif xu <= 0:
            return (wu, 0), (wl, 0)
    # A --- B 
    # |     |
    # C --- D
    A = xl, max(xl*wl, xl*wu)
    B = xu, max(xu*wl, xu*wu)
    C = xl, min(xl*wl, xl*wu)
    D = xu, min(xu*wl, xu*wu)
    def getLine(ST, ED):
        k = (ED[1] - ST[1]) / (ED[0] - ST[0])
        b = ST[1] - k * ST[0]
        return k, b
    k1, b1 = getLine(C, D)
    k2, b2 = getLine(A, B)
    return (k1, b1), (k2, b2)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def d_sigmoid(z):
    y = sigmoid(z)
    return y * (1 - y)

def slope_point_repre(k, x, y):
    """ return (k,b) that y = kx + b """
    return k, y - k * x

def sigmoid_abstract(wl, wu, xl, xu):
    """ compute the two-line abstract that bounds 
            k1 * x + b1 <= sigmoid(wx) <= k2 * x + b2 
    where w and x are intervals that might cross the origin
    """
    assert wl <= wu and xl <= xu, "must wl <= wu and xl <= xu"
    assert wl >= 0 or wu <= 0, "must wl >= 0 or wu <= 0 (ie. do not cross origin)"
    
    if xl == xu:
        x = xl
        return (0, wl * sigmoid(x)), (0, wu * sigmoid(x))

    floating_point_safeguard = 1e-5
    xl -= floating_point_safeguard
    xu += floating_point_safeguard
    
    gl,gu = sigmoid(xl), sigmoid(xu)
    gl_prime, gu_prime = d_sigmoid(xl), d_sigmoid(xu)
    kappa = (gu - gl) / (xu - xl)
    kappa_prime = min(gl_prime, gu_prime)

    if xl >= 0:
        if wl >= 0:
            return slope_point_repre(k=wl*kappa, x=xl, y=wl*gl), \
                   slope_point_repre(k=wu*kappa_prime, x=xu, y=wu*gu)
        elif wu <= 0:
            return slope_point_repre(k=wl*kappa_prime, x=xu, y=wl*gu), \
                   slope_point_repre(k=wu*kappa, x=xl, y=wu*gl)
    elif xu <= 0:
        if wl >= 0:
            return slope_point_repre(k=wl*kappa_prime, x=xl, y=wl*gl), \
                   slope_point_repre(k=wu*kappa, x=xu, y=wu*gu)
        elif wu <= 0:
            return slope_point_repre(k=wl*kappa, x=xu, y=wl*gu), \
                   slope_point_repre(k=wu*kappa_prime, x=xl, y=wu*gl)
    else: # xl < 0 < xu
        if wl >= 0:
            return slope_point_repre(k=wl*kappa_prime, x=xl, y=wl*gl), \
                   slope_point_repre(k=wu*kappa_prime, x=xu, y=wu*gu)
        elif wu <= 0:
            return slope_point_repre(k=wl*kappa_prime, x=xu, y=wl*gu), \
                   slope_point_repre(k=wu*kappa_prime, x=xl, y=wu*gl)

    raise NotImplementedError("Implemented but impossible branch case!")

def tanh(z):
    e_x = np.exp(z)
    e_nx = np.exp(-z)
    return (e_x - e_nx) / (e_x + e_nx)

def d_tanh(z):
    y = tanh(z)
    return 1 - y ** 2

def tanh_abstract(wl, wu, xl, xu):
    """ compute the two-line abstract that bounds
            k1 * x + b1 <= tanh(wx) <= k2 * x + b2
        where w and x are intervals that might cross the origin
    """
    assert wl <= wu and xl <= xu, "must wl <= wu and xl <= xu"
    assert wl >= 0 or wu <= 0, "must wl >= 0 or wu <= 0 (ie. do not cross origin)"
    
    if xl == xu:
        x = xl
        val = tanh(x)
        comb1, comb2 = wl * val, wu * val
        
        return (0, min(comb1, comb2)), (0, max(comb1, comb2))

    floating_point_safeguard = 1e-5
    xl -= floating_point_safeguard
    xu += floating_point_safeguard
    
    gl,gu = tanh(xl), tanh(xu)
    gl_prime, gu_prime = d_tanh(xl), d_tanh(xu)
    kappa = (gu - gl) / (xu - xl)
    kappa_prime = min(gl_prime, gu_prime)
    
    if xl >= 0:
        if wl >= 0:
            return slope_point_repre(k=wl*kappa, x=xl, y=wl*gl), \
                   slope_point_repre(k=wu*kappa_prime, x=xu, y=wu*gu)
        elif wu <= 0:
            return slope_point_repre(k=wl*kappa_prime, x=xu, y=wl*gu), \
                   slope_point_repre(k=wu*kappa, x=xl, y=wu*gl)
    elif xu <= 0:
        if wl >= 0:
            return slope_point_repre(k=wu*kappa_prime, x=xl, y=wu*gl), \
                   slope_point_repre(k=wl*kappa, x=xu, y=wl*gu)
        elif wu <= 0:
            return slope_point_repre(k=wu*kappa, x=xu, y=wu*gu), \
                   slope_point_repre(k=wl*kappa_prime, x=xl, y=wl*gl)
    else: # xl < 0 < xu
        if wl >= 0:
            return slope_point_repre(k=wl*kappa_prime, x=xl, y=wu*gl), \
                   slope_point_repre(k=wl*kappa_prime, x=xu, y=wu*gu)
        elif wu <= 0:
            return slope_point_repre(k=wu*kappa_prime, x=xu, y=wl*gu), \
                   slope_point_repre(k=wu*kappa_prime, x=xl, y=wl*gl)
    
    raise NotImplementedError("Implemented but impossible branch case!")

class DP_DNN_neuron(object):

    def __init__(self):
        self.algebra_lower = None
        self.algebra_upper = None
        self.concrete_algebra_lower = None
        self.concrete_algebra_upper = None
        self.concrete_lower = None
        self.concrete_lower_noClip = None
        self.concrete_upper = None
        self.concrete_upper_noClip = NotImplementedError
        self.concrete_highest_lower = None
        self.concrete_lowest_upper = None
        self.historical_concrete_lowest_lower = None
        self.historical_concrete_highest_upper = None
        self.weight = None
        self.weight_deepcopy = None
        self.bias = None
        self.prev_abs_mode = None
        self.prev_abs_mode_min = None
        self.certain_flag = 0
        self.variable_weight = False
        self.variable_weight_id = None # this activation is relu(w * x[id]) where w \in [rangeMin, rangeMax]
        self.variable_weight_min = None
        self.variable_weight_max = None
        self.actMode = 0 # 1: activated ; 2: deactivated; 3: lb+ub>=0; 4: lb+ub<0

    def clear(self):
        self.certain_flag = 0
        self.actMode = 0
        self.concrete_highest_lower = None
        self.concrete_lowest_upper = None
        self.prev_abs_mode = None
        self.variable_weight = False
        self.variable_weight_id = None
        self.variable_weight_min = None
        self.variable_weight_max = None
        self.variable_weight_min_abs = None
        self.variable_weight_max_abs = None
        self.historical_concrete_lowest_lower = None
        self.historical_concrete_highest_upper = None
    
    def clear_weight_range(self,hintWeightIndex = None):
        if self.variable_weight:
            if hintWeightIndex != None:
                self.weight[hintWeightIndex] = deepcopy(self.weight_deepcopy[hintWeightIndex])
            else:
                self.weight = deepcopy(self.weight_deepcopy)
            self.variable_weight = False
            self.variable_weight_id = None
            self.variable_weight_min = None
            self.variable_weight_max = None
            self.variable_weight_min_abs = None
            self.variable_weight_max_abs = None

    def print_detail(cur_neuron, i, j):
            print("Layer, neuron: ", i + 1, j)
            print("Algebra Lower: ", cur_neuron.algebra_lower)
            print("Algebra Upper: ", cur_neuron.algebra_upper)
            print("Concrete Lower: ", cur_neuron.concrete_lower)
            print("Concrete Upper: ", cur_neuron.concrete_upper)
            print("Act Mode: ", ["affine mode",
                                    "pre_neuron.concrete_highest_lower >= 0", 
                                    "pre_neuron.concrete_lowest_upper < 0",
                                    "pre_neuron.concrete_highest_lower + pre_neuron.concrete_lowest_upper < 0",
                                    "pre_neuron.concrete_highest_lower + pre_neuron.concrete_lowest_upper >= 0"][cur_neuron.actMode]) 
            print("Concrete Algebra Lower: ", cur_neuron.concrete_algebra_lower)
            print("Concrete Algebra Upper: ", cur_neuron.concrete_algebra_upper)
            print('-----')

class DP_DNN_layer(object):
    INPUT_LAYER = 0
    AFFINE_LAYER = 1
    RELU_LAYER = 2
    SIGMOID_LAYER = 3
    TANH_LAYER = 4

    def __init__(self):
        self.size = None
        self.size_deepcopy = None
        self.neurons = None
        self.layer_type = None
        self.DeltaW = None # the quantified scaling factor
        self.auxLayer = False # symbols if an affine layer is an auxiliary layer

    def clear(self):
        for i in range(len(self.neurons)):
            self.neurons[i].clear()
        raise NotImplementedError("The clear function is not implemented.")

    def clear_weight_range(self,hintNeuronIndex=None,hintWeightIndex=None):
        if hintNeuronIndex != None:
            self.neurons[hintNeuronIndex].clear_weight_range(hintWeightIndex)
        else:
            for i in range(len(self.neurons)):
                self.neurons[i].clear_weight_range()
        # self.neurons.resize(self.size_deepcopy)
        self.neurons = self.neurons[:self.size_deepcopy]
        self.size = self.size_deepcopy + 0
    
    def clear_virtual(self):
        self.neurons = self.neurons[:self.size_deepcopy]
        self.size = self.size_deepcopy + 0 

class DP_DNN_network(object):

    def __init__(self, ifSignedOutput):
        self.MODE_QUANTITIVE = 0
        self.MODE_ROBUSTNESS = 1

        self.numlayers = None
        self.layerSizes = None
        self.inputSize = None
        self.outputSize = None
        self.mins = None
        self.maxes = None
        self.ranges = None
        self.layers = None
        self.property_flag = None
        self.property_region = None
        self.abs_mode_changed = None
        self.reluN = 6
        self.outSigned = True
        self.outputSigned = ifSignedOutput
        self.weight_variables = {}
        #(first layer)  self.weight_variables[(trueLayerIndex,neuronIndex)] = [(weightIndex,rangeMin,rangeMax)...]
        #               self.weight_variables[(trueLayerIndex,neuronIndex,weightIndex)] = [(rangeMin,rangeMax)]
        self.bias_variables = {}
        # self.bias_variables[(trueLayerIndex,neuronIndex)] = (rangeMin,rangeMax)


    def clear(self):
        for i in range(len(self.layers)):
            self.layers[i].clear()
        self.weight_variables = {}
        raise NotImplementedError("The clear function is not implemented.")
        # function need to be done: remove the virtual neurons
    def clear_weight_range(self):
        """
        Clear all the weight range variables. Preserve the network. 
        """
        for _key in self.weight_variables:
            if len(_key) == 2:
                continue
            trueLayerIndex,neuronIndex,weightIndex = _key
            assert trueLayerIndex > 0
            self.layers[trueLayerIndex].clear_weight_range(neuronIndex,weightIndex)
            self.layers[trueLayerIndex - 1].clear_virtual()

        self.weight_variables = {}
    
    def deeppoly(self, VERBOSE = False):
        
        def reluw_abstract(cur_neuron:DP_DNN_neuron):
            assert cur_neuron.weight == None #kind of assersion of a relu layer
            if cur_neuron.variable_weight:
                if cur_neuron.variable_weight_min >= 0: 
                    if cur_neuron.concrete_lower is not None:
                        cur_neuron.concrete_lower, cur_neuron.concrete_upper = cur_neuron.variable_weight_min * cur_neuron.concrete_lower,cur_neuron.variable_weight_max * cur_neuron.concrete_upper
                    if cur_neuron.concrete_highest_lower is not None:
                        cur_neuron.concrete_highest_lower, cur_neuron.concrete_lowest_upper = cur_neuron.variable_weight_min * cur_neuron.concrete_highest_lower, cur_neuron.variable_weight_max * cur_neuron.concrete_lowest_upper
                    cur_neuron.algebra_lower, cur_neuron.algebra_upper = cur_neuron.variable_weight_min * cur_neuron.algebra_lower, cur_neuron.variable_weight_max * cur_neuron.algebra_upper
                    if cur_neuron.concrete_algebra_lower is not None:
                        cur_neuron.concrete_algebra_lower, cur_neuron.concrete_algebra_upper = cur_neuron.concrete_algebra_lower * cur_neuron.variable_weight_min, cur_neuron.concrete_algebra_upper * cur_neuron.variable_weight_max
                if cur_neuron.variable_weight_max <= 0:
                    if cur_neuron.concrete_lower is not None:
                        cur_neuron.concrete_lower, cur_neuron.concrete_upper = cur_neuron.variable_weight_min * cur_neuron.concrete_upper, cur_neuron.variable_weight_max * cur_neuron.concrete_lower
                    if cur_neuron.concrete_highest_lower is not None:
                        cur_neuron.concrete_highest_lower, cur_neuron.concrete_lowest_upper = cur_neuron.variable_weight_min * cur_neuron.concrete_lowest_upper, cur_neuron.variable_weight_max * cur_neuron.concrete_highest_lower
                    cur_neuron.algebra_lower, cur_neuron.algebra_upper = cur_neuron.variable_weight_min * cur_neuron.algebra_upper, cur_neuron.variable_weight_max * cur_neuron.algebra_lower
                    if cur_neuron.concrete_algebra_lower is not None:
                        cur_neuron.concrete_algebra_lower, cur_neuron.concrete_algebra_upper = cur_neuron.concrete_algebra_upper * cur_neuron.variable_weight_min, cur_neuron.concrete_algebra_lower * cur_neuron.variable_weight_max
                elif cur_neuron.variable_weight_min < 0 and 0 < cur_neuron.variable_weight_max: 
                    if cur_neuron.concrete_lower is not None:
                        cur_neuron.concrete_lower, cur_neuron.concrete_upper = cur_neuron.variable_weight_min * cur_neuron.concrete_upper, cur_neuron.variable_weight_max * cur_neuron.concrete_upper
                    if cur_neuron.concrete_highest_lower is not None:
                        cur_neuron.concrete_highest_lower, cur_neuron.concrete_lowest_upper = cur_neuron.variable_weight_min * cur_neuron.concrete_lowest_upper, cur_neuron.variable_weight_max * cur_neuron.concrete_lowest_upper
                    cur_neuron.algebra_lower, cur_neuron.algebra_upper = cur_neuron.variable_weight_min * cur_neuron.algebra_upper, cur_neuron.variable_weight_max * cur_neuron.algebra_upper
                    if cur_neuron.concrete_algebra_lower is not None:
                        cur_neuron.concrete_algebra_lower, cur_neuron.concrete_algebra_upper = cur_neuron.concrete_algebra_upper * cur_neuron.variable_weight_min, cur_neuron.concrete_algebra_upper * cur_neuron.variable_weight_max
                if VERBOSE:
                    cur_neuron.print_detail(-1, "ReluW Abstract")
        def pre_gpu(cur_neuron, i):
            if i == 0:
                cur_neuron.concrete_algebra_lower = deepcopy(cur_neuron.algebra_lower)
                cur_neuron.concrete_algebra_upper = deepcopy(cur_neuron.algebra_upper)
            lower_bound = torch.tensor(deepcopy(cur_neuron.algebra_lower))
            upper_bound = torch.tensor(deepcopy(cur_neuron.algebra_upper))
            for k in range(i + 1)[::-1]:
                tmp_lower = torch.zeros(len(self.layers[k].neurons[0].algebra_lower)).cuda(non_blocking=True)
                tmp_upper = torch.zeros(len(self.layers[k].neurons[0].algebra_lower)).cuda(non_blocking=True)
                assert (self.layers[k].size + 1 == len(lower_bound))
                assert (self.layers[k].size + 1 == len(upper_bound))
                for p in range(self.layers[k].size):
                    if lower_bound[p] >= 0:
                        tmp_lower += lower_bound[p] * torch.tensor(self.layers[k].neurons[p].algebra_lower).cuda()
                    else:
                        tmp_lower += lower_bound[p] * torch.tensor(self.layers[k].neurons[p].algebra_upper).cuda()

                    if upper_bound[p] >= 0:
                        tmp_upper += upper_bound[p] * torch.tensor(self.layers[k].neurons[p].algebra_upper).cuda()
                    else:
                        tmp_upper += upper_bound[p] * torch.tensor(self.layers[k].neurons[p].algebra_lower).cuda()
                tmp_lower[-1] += lower_bound[-1]
                tmp_upper[-1] += upper_bound[-1]
                lower_bound = tmp_lower.clone()
                upper_bound = tmp_upper.clone()
                if k == 1:
                    cur_neuron.concrete_algebra_upper = deepcopy(upper_bound.cpu().numpy())
                    cur_neuron.concrete_algebra_lower = deepcopy(lower_bound.cpu().numpy())
            assert (len(upper_bound.cpu().numpy()) == 1)
            assert (len(lower_bound.cpu().numpy()) == 1)
            cur_neuron.concrete_lower = lower_bound[0].item()
            cur_neuron.concrete_upper = upper_bound[0].item()
            #discard the concrete value from the previous deeppolys
            cur_neuron.concrete_highest_lower = None
            cur_neuron.concrete_lowest_upper = None
            if (cur_neuron.concrete_highest_lower == None) or (
                    cur_neuron.concrete_highest_lower < cur_neuron.concrete_lower):
                cur_neuron.concrete_highest_lower = cur_neuron.concrete_lower
            if (cur_neuron.concrete_lowest_upper == None) or (
                    cur_neuron.concrete_lowest_upper > cur_neuron.concrete_upper):
                cur_neuron.concrete_lowest_upper = cur_neuron.concrete_upper
                

        def pre(cur_neuron, i):
            # return pre_gpu(cur_neuron, i)
            if i == 0:
                cur_neuron.concrete_algebra_lower = deepcopy(cur_neuron.algebra_lower)
                cur_neuron.concrete_algebra_upper = deepcopy(cur_neuron.algebra_upper)
            lower_bound = deepcopy(cur_neuron.algebra_lower)
            upper_bound = deepcopy(cur_neuron.algebra_upper)
            for k in range(i + 1)[::-1]:
                tmp_lower = np.zeros(len(self.layers[k].neurons[0].algebra_lower))
                tmp_upper = np.zeros(len(self.layers[k].neurons[0].algebra_lower))
                assert (self.layers[k].size + 1 == len(lower_bound))
                assert (self.layers[k].size + 1 == len(upper_bound))
                for p in range(self.layers[k].size):
                    if lower_bound[p] >= 0:
                        tmp_lower += lower_bound[p] * self.layers[k].neurons[p].algebra_lower
                    else:
                        tmp_lower += lower_bound[p] * self.layers[k].neurons[p].algebra_upper

                    if upper_bound[p] >= 0:
                        tmp_upper += upper_bound[p] * self.layers[k].neurons[p].algebra_upper
                    else:
                        tmp_upper += upper_bound[p] * self.layers[k].neurons[p].algebra_lower
                # print("tmp_upper =", tmp_upper)
                tmp_lower[-1] += lower_bound[-1]
                tmp_upper[-1] += upper_bound[-1]
                lower_bound = deepcopy(tmp_lower)
                upper_bound = deepcopy(tmp_upper)
                if k == 1:
                    cur_neuron.concrete_algebra_upper = deepcopy(upper_bound)
                    cur_neuron.concrete_algebra_lower = deepcopy(lower_bound)
            assert (len(lower_bound) == 1)
            assert (len(upper_bound) == 1)
            cur_neuron.concrete_lower = lower_bound[0]
            cur_neuron.concrete_upper = upper_bound[0]
            #discard the concrete value from the previous deeppolys
            cur_neuron.concrete_highest_lower = None
            cur_neuron.concrete_lowest_upper = None
            if (cur_neuron.concrete_highest_lower == None) or (
                    cur_neuron.concrete_highest_lower < cur_neuron.concrete_lower):
                cur_neuron.concrete_highest_lower = cur_neuron.concrete_lower
            if (cur_neuron.concrete_lowest_upper == None) or (
                    cur_neuron.concrete_lowest_upper > cur_neuron.concrete_upper):
                cur_neuron.concrete_lowest_upper = cur_neuron.concrete_upper
            if cur_neuron.historical_concrete_lowest_lower == None:
                cur_neuron.historical_concrete_lowest_lower = cur_neuron.concrete_lower
            if cur_neuron.historical_concrete_highest_upper == None:
                cur_neuron.historical_concrete_highest_upper = cur_neuron.concrete_upper
            cur_neuron.historical_concrete_lowest_lower = min(cur_neuron.historical_concrete_lowest_lower, 
                                                            cur_neuron.concrete_lower)
            cur_neuron.historical_concrete_highest_upper = max(cur_neuron.historical_concrete_highest_upper,
                                                            cur_neuron.concrete_upper)

        self.abs_mode_changed = 0
        self.abs_mode_changed_min = 0
        if VERBOSE:
            i = 0
            for j in range(len(self.layers[i].neurons)):
                self.layers[i].neurons[j].print_detail(i - 1, j)
        for i in range(len(self.layers) - 1):
            gp_layer_count = 0
            pre_layer = self.layers[i]
            cur_layer = self.layers[i + 1]
            pre_neuron_list = pre_layer.neurons
            cur_neuron_list = cur_layer.neurons

            # if i == (len(self.layers)-2):
            #     print("Jump here")
            if cur_layer.layer_type == DP_DNN_layer.AFFINE_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron = cur_neuron_list[j]
                    # test code
                    # test_weight = cur_neuron.weight
                    # test_bias = cur_neuron.bias
                    # print("Weight's DType: ", test_weight.dtype)
                    # print("test_bias's DType: ", test_weight.dtype)
                    cur_neuron.algebra_lower = np.append(cur_neuron.weight, [0])
                    cur_neuron.algebra_upper = np.append(cur_neuron.weight, [0])
                    if (i + 1, j) in self.weight_variables:
                        for (weightIndex, rangeMin, rangeMax) in self.weight_variables[(i + 1, j)]:
                            (k1, b1), (k2, b2) = weighted_input_relax(
                                rangeMin, rangeMax, pre_neuron_list[weightIndex].concrete_lower, pre_neuron_list[weightIndex].concrete_upper
                            )
                            cur_neuron.algebra_lower[weightIndex] = k1
                            cur_neuron.algebra_upper[weightIndex] = k2
                            cur_neuron.algebra_lower[-1] += b1
                            cur_neuron.algebra_upper[-1] += b2                            
                            
                    if (i + 1, j) in self.bias_variables:
                        rangeMin, rangeMax = self.bias_variables[(i + 1, j)]
                        cur_neuron.algebra_lower[-1] += rangeMin
                        cur_neuron.algebra_upper[-1] += rangeMax
                    else:
                        cur_neuron.algebra_lower[-1] += cur_neuron.bias
                        cur_neuron.algebra_upper[-1] += cur_neuron.bias

                    pre(cur_neuron, i)

                    cur_neuron.concrete_lower_noClip = cur_neuron.concrete_lower
                    cur_neuron.concrete_upper_noClip = cur_neuron.concrete_upper
                    cur_neuron.actMode = 0 # affine mode
                    if VERBOSE:
                        cur_neuron.print_detail(i, j)
                gp_layer_count = gp_layer_count + 1

            elif cur_layer.layer_type == DP_DNN_layer.RELU_LAYER:
                for _j in range(cur_layer.size):
                    cur_neuron : DP_DNN_neuron = cur_neuron_list[_j]
                    if cur_neuron.variable_weight:
                        pre_neuron = pre_neuron_list[cur_neuron.variable_weight_id]
                        j = cur_neuron.variable_weight_id
                    else:
                        pre_neuron = pre_neuron_list[_j]
                        j = _j

                    cur_neuron.concrete_lower_noClip = pre_neuron.concrete_lower_noClip
                    cur_neuron.concrete_upper_noClip = pre_neuron.concrete_upper_noClip
                    if pre_neuron.concrete_highest_lower >= 0 or cur_neuron.certain_flag == 1:
                        cur_neuron.algebra_lower = np.zeros(pre_layer.size + 1)
                        cur_neuron.algebra_upper = np.zeros(pre_layer.size + 1)
                        cur_neuron.algebra_lower[j] = 1
                        cur_neuron.algebra_upper[j] = 1
                        cur_neuron.concrete_algebra_lower = deepcopy(pre_neuron.concrete_algebra_lower)
                        cur_neuron.concrete_algebra_upper = deepcopy(pre_neuron.concrete_algebra_upper)
                        cur_neuron.concrete_lower = pre_neuron.concrete_lower
                        cur_neuron.concrete_upper = pre_neuron.concrete_upper
                        cur_neuron.concrete_highest_lower = pre_neuron.concrete_highest_lower
                        cur_neuron.concrete_lowest_upper = pre_neuron.concrete_lowest_upper
                            
                        #compute abstract domain
                        reluw_abstract(cur_neuron)
                        
                        # cur_neuron.certain_flag = 1
                        cur_neuron.actMode = 1
                    elif pre_neuron.concrete_lowest_upper < 0 or cur_neuron.certain_flag == 2:
                        cur_neuron.algebra_lower = np.zeros(pre_layer.size + 1)
                        cur_neuron.algebra_upper = np.zeros(pre_layer.size + 1)
                        cur_neuron.concrete_algebra_lower = np.zeros(self.inputSize + 1)
                        cur_neuron.concrete_algebra_upper = np.zeros(self.inputSize + 1)
                        cur_neuron.concrete_lower = 0
                        cur_neuron.concrete_upper = 0
                        cur_neuron.concrete_highest_lower = 0
                        cur_neuron.concrete_lowest_upper = 0
                        # cur_neuron.certain_flag = 2
                        cur_neuron.actMode = 2
                    elif pre_neuron.concrete_highest_lower + pre_neuron.concrete_lowest_upper < 0:
                        if (cur_neuron.prev_abs_mode != None) and (cur_neuron.prev_abs_mode != 0):
                            self.abs_mode_changed += 1
                        cur_neuron.prev_abs_mode = 0

                        cur_neuron.algebra_lower = np.zeros(pre_layer.size + 1)
                        cur_neuron.algebra_upper = np.zeros(pre_layer.size + 1)

                        aux = pre_neuron.concrete_lowest_upper / (
                                pre_neuron.concrete_lowest_upper - pre_neuron.concrete_highest_lower)
                        
                        
                        cur_neuron.algebra_upper[j] = aux
                        cur_neuron.algebra_upper[-1] = -aux * pre_neuron.concrete_highest_lower
                        
                        #compute abstract domain
                        reluw_abstract(cur_neuron)
                        
                        pre(cur_neuron, i)
                        cur_neuron.actMode = 3
                    else:
                        if (cur_neuron.prev_abs_mode != None) and (cur_neuron.prev_abs_mode != 1):
                            self.abs_mode_changed += 1
                        cur_neuron.prev_abs_mode = 1

                        cur_neuron.algebra_lower = np.zeros(pre_layer.size + 1)
                        cur_neuron.algebra_upper = np.zeros(pre_layer.size + 1)

                        cur_neuron.algebra_lower[j] = 1

                        aux = pre_neuron.concrete_lowest_upper / (
                                pre_neuron.concrete_lowest_upper - pre_neuron.concrete_highest_lower)

                        cur_neuron.algebra_upper[j] = aux
                        cur_neuron.algebra_upper[-1] = -aux * pre_neuron.concrete_highest_lower
                        
                        #compute abstract domain
                        reluw_abstract(cur_neuron)

                        pre(cur_neuron, i)
                        cur_neuron.actMode = 4
                    if VERBOSE:
                        cur_neuron.print_detail(i, _j)
            elif cur_layer.layer_type in [DP_DNN_layer.SIGMOID_LAYER, DP_DNN_layer.TANH_LAYER]:
                actvFac = []
                actvCst = []
                for _j in range(cur_layer.size):
                    cur_neuron : DP_DNN_neuron = cur_neuron_list[_j]
                    if cur_neuron.variable_weight:
                        pre_neuron = pre_neuron_list[cur_neuron.variable_weight_id]
                        j = cur_neuron.variable_weight_id
                        rangeMin, rangeMax = cur_neuron.variable_weight_min, cur_neuron.variable_weight_max
                    else:
                        pre_neuron = pre_neuron_list[_j]
                        j = _j
                        rangeMin, rangeMax = 1, 1
                    
                    cur_neuron.algebra_lower = np.zeros(pre_layer.size + 1)
                    cur_neuron.algebra_upper = np.zeros(pre_layer.size + 1)
                    
                    if cur_layer.layer_type == DP_DNN_layer.SIGMOID_LAYER:
                        (k1, b1), (k2, b2) = sigmoid_abstract(
                            wl = rangeMin, wu = rangeMax, xl=pre_neuron.concrete_lower, xu=pre_neuron.concrete_upper
                        )
                    elif cur_layer.layer_type == DP_DNN_layer.TANH_LAYER:
                        (k1, b1), (k2, b2) = tanh_abstract(
                            wl = rangeMin, wu = rangeMax, xl=pre_neuron.concrete_lower, xu=pre_neuron.concrete_upper
                        )

                    cur_neuron.algebra_lower[j] = k1
                    cur_neuron.algebra_upper[j] = k2
                    cur_neuron.algebra_lower[-1] = b1
                    cur_neuron.algebra_upper[-1] = b2
                    
                    actvFac.append((k1, k2))
                    actvCst.append((b1, b2))
                    
                    pre(cur_neuron, i)
                    if VERBOSE:
                        cur_neuron.print_detail(i, _j)
                # print(f"actvFac: {actvFac}")
                # print(f"actvCst: {actvCst}")

                    
    def add_difference_layer(self,outputLabel):
        """
        Add a difference layer to the network. 
        The outputLabel is the label of the output neuron that should be the difference of the two neurons. 
        newlayer consists output[outputLabel] - output[i]
        """
        if outputLabel >= self.layers[-1].size:
            raise ValueError("The output label is out of range.")
        
        newLayer = DP_DNN_layer()
        newLayer.layer_type = DP_DNN_layer.AFFINE_LAYER
        newLayer.size = self.layers[-1].size
        newLayer.neurons = []
        for i in range(newLayer.size):
            newNeuron = DP_DNN_neuron()
            newNeuron.weight = np.zeros(newLayer.size)
            newNeuron.bias = 0
            newNeuron.weight[outputLabel] = 1
            newNeuron.weight[i] = -1
            newLayer.neurons.append(newNeuron)
        newLayer.auxLayer = True
        self.layers.append(newLayer)
        self.layerSizes.append(newLayer.size)
    
    def create_network(self,weights,bias):
        """
        create relu network from weights
        weight = [
            [
                [],[],[],[]
            ]
        ]
        bias = [
            [
                b,b,b,b
            ]
        ]
        insert relu to layers except for the first and the last layer
        """
        self.layers = []
        input = DP_DNN_layer()
        input.layer_type = DP_DNN_layer.INPUT_LAYER
        input.size = len(weights[0])
        input.neurons = [DP_DNN_neuron() for i in range(input.size)]
        self.layers.append(input)
        self.layerSizes = []
        self.layerSizes.append(input.size)
        for (id,wblk) in enumerate(weights[1:]):
            affine = DP_DNN_layer()
            affine.layer_type = DP_DNN_layer.AFFINE_LAYER
            affine.size = len(wblk)
            affine.neurons = []
            for (id2,w) in enumerate(wblk):
                new_neuron = DP_DNN_neuron()
                new_neuron.weight = np.array(w)
                new_neuron.bias = bias[id][id2]
                affine.neurons.append(new_neuron)
            self.layers.append(affine)
            self.layerSizes.append(affine.size)
            if id != len(weights) - 2:
                relu = DP_DNN_layer()
                relu.layer_type = DP_DNN_layer.RELU_LAYER
                relu.size = len(wblk)
                relu.neurons = []
                for i in range(len(wblk)):
                    new_neuron = DP_DNN_neuron()
                    relu.neurons.append(new_neuron)
                self.layers.append(relu)
                self.layerSizes.append(relu.size)
        for layer in self.layers:
            layer.size_deepcopy = layer.size
        self.inputSize = self.layerSizes[0]

    def load_dnn(self, quantized_model):

        layersize = []
        self.layers = []

        layersize.append(quantized_model._input_shape[-1])

        new_in_layer = DP_DNN_layer()
        new_in_layer.layer_type = DP_DNN_layer.INPUT_LAYER
        new_in_layer.size = layersize[-1]
        new_in_layer.neurons = []
        for i in range(layersize[-1]):
            new_neuron = DP_DNN_neuron()
            new_in_layer.neurons.append(new_neuron)
        self.layers.append(new_in_layer)

        numDensLayers = len(quantized_model.dense_layers)
        for i, l in enumerate(quantized_model.dense_layers):
            tf_layer = quantized_model.dense_layers[i]
            w, b = tf_layer.get_weights()
            w = w.T
            layersize.append(l.units)
            if (i < numDensLayers - 1):
                new_hidden_layer = DP_DNN_layer()
                new_hidden_layer.layer_type = DP_DNN_layer.AFFINE_LAYER
                new_hidden_layer.size = layersize[-1]
                new_hidden_layer.neurons = []
                for k in range(layersize[-1]):
                    new_hidden_neuron = DP_DNN_neuron()
                    new_hidden_neuron.weight = w[k]
                    new_hidden_neuron.bias = b[k]
                    new_hidden_layer.neurons.append(new_hidden_neuron)
                self.layers.append(new_hidden_layer)

                new_hidden_layer = DP_DNN_layer()
                new_hidden_layer.layer_type = DP_DNN_layer.RELU_LAYER
                new_hidden_layer.size = layersize[-1]
                new_hidden_layer.neurons = []
                for k in range(layersize[-1]):
                    new_hidden_neuron = DP_DNN_neuron()
                    new_hidden_layer.neurons.append(new_hidden_neuron)
                self.layers.append(new_hidden_layer)

            else:
                new_out_layer = DP_DNN_layer()
                new_out_layer.layer_type = new_out_layer.AFFINE_LAYER
                new_out_layer.size = layersize[-1]
                new_out_layer.neurons = []
                for k in range(layersize[-1]):
                    new_out_neuron = DP_DNN_neuron()
                    new_out_neuron.weight = w[k]
                    new_out_neuron.bias = b[k]
                    new_out_layer.neurons.append(new_out_neuron)
                self.layers.append(new_out_layer)

        for layer in self.layers:
            layer.size_deepcopy = layer.size

        self.layerSizes = layersize
        self.inputSize = layersize[0]
        self.outputSize = layersize[-1]
        self.numLayers = len(layersize) - 1
    
    def change_activation(self, activation:str):
        layer_code = None
        if activation == "relu":
            layer_code = DP_DNN_layer.RELU_LAYER
        elif activation == "sigmoid":
            layer_code = DP_DNN_layer.SIGMOID_LAYER
        elif activation == "tanh":
            layer_code = DP_DNN_layer.TANH_LAYER

        for layer in self.layers:
            layer :DP_DNN_layer = layer
            if layer.layer_type in [DP_DNN_layer.RELU_LAYER, DP_DNN_layer.SIGMOID_LAYER, DP_DNN_layer.TANH_LAYER]:
                layer.layer_type = layer_code 

    def return_output(self):
        return [(neuron.concrete_lower, neuron.concrete_upper) for neuron in self.layers[-1].neurons]

    def add_weight_range(self, affineLayerIndex, neuronIndex, weightIndex, rangeMin, rangeMax):
            """
            Change affine_layer[affineLayerIndex].neuron[nenuronIndex].weight[weightIndex] to rangeMin~rangeMax.
            Set affine_layer[affineLayerIndex].neuron[nenuronIndex].weight[weightIndex] to 0 first. 
            Add a new virtual neuron at layer (affineLayerIndex - 1) (which would be a RELU layer).
            Set the special abstract domain of the new virtual neuron parameter rangeMin rangeMax.
            Assert that rangeMin <= rangeMax and the original weight is in the range. 
            """
            if rangeMin > rangeMax:
                raise ValueError("The rangeMin should be equal or less than rangeMax")

            #find the true index of the affineLayerIndex
            affinLayerCount = 0
            trueLayerIndex = None
            for (idx,layer) in enumerate(self.layers):
                if layer.layer_type == DP_DNN_layer.AFFINE_LAYER:
                    affinLayerCount += 1
                    if affineLayerIndex == affinLayerCount:
                        trueLayerIndex = idx
                        break 
            if trueLayerIndex == None:
                raise ValueError("The affineLayerIndex is out of range")

            #quote the layer and the neuron
            layer : DP_DNN_layer = self.layers[trueLayerIndex]
            neuron : DP_DNN_neuron = layer.neurons[neuronIndex]
            if neuron.weight_deepcopy is None:
                neuron.weight_deepcopy = deepcopy(neuron.weight)
            weight_deepcopy = neuron.weight_deepcopy[weightIndex]
            thePreviousLayer: DP_DNN_layer = self.layers[trueLayerIndex-1]

            if (trueLayerIndex,neuronIndex,weightIndex) in self.weight_variables:
                raise ValueError("This edge ({}) has been added for some weight!, which is {}".format(
                        (trueLayerIndex,neuronIndex,weightIndex),
                        self.weight_variables[(trueLayerIndex,neuronIndex,weightIndex)]
                    )
                )
    
            if thePreviousLayer.layer_type not in [DP_DNN_layer.RELU_LAYER, DP_DNN_layer.SIGMOID_LAYER, DP_DNN_layer.TANH_LAYER]:
                if thePreviousLayer.layer_type == DP_DNN_layer.INPUT_LAYER:
                    if (trueLayerIndex,neuronIndex) in self.weight_variables:
                        self.weight_variables[(trueLayerIndex,neuronIndex)].append((weightIndex,rangeMin,rangeMax))
                    else:
                        self.weight_variables[(trueLayerIndex,neuronIndex)] = [(weightIndex,rangeMin,rangeMax)]
                    return 
                else:
                    raise ValueError("The previous layer is neither a RELU layer or an INPUT layer.")

            # if not (rangeMin <= weight_deepcopy and weight_deepcopy <= rangeMax):
            #     raise ValueError(f"The original weight is not in the range {weight_deepcopy} [{rangeMin},{rangeMax}]")

            # if rangeMin * rangeMax < 0:
            #     raise ValueError("The range should not cross the 0")
            
            # if rangeMin == rangeMax:
            #     raise ValueError("The range should not be a point")

            # add the new virtual neural
            
            
            newVirtualNeuron = DP_DNN_neuron()
            newVirtualNeuron.variable_weight = True
            newVirtualNeuron.variable_weight_id = weightIndex
            newVirtualNeuron.variable_weight_min = rangeMin
            newVirtualNeuron.variable_weight_max = rangeMax

            # newVirtualNeuron.variable_weight_min_abs = min(abs(rangeMin),abs(rangeMax))
            # newVirtualNeuron.variable_weight_max_abs = max(abs(rangeMin),abs(rangeMax))

            thePreviousLayer.neurons.append(newVirtualNeuron)
            thePreviousLayer.size += 1

            # change the weights array of all neurals in this layer in accordance with the modified previous layer
            for neuron_i in self.layers[trueLayerIndex].neurons:
                neuron_i.weight = np.append(neuron_i.weight, [0])

            # change the current neuron's weight to the new virtual neural
            neuron.weight[weightIndex] = 0
            neuron.weight[-1] = 1 # if (rangeMin >= 0 and rangeMax >= 0) else -1

            self.weight_variables[(trueLayerIndex,neuronIndex,weightIndex)] = (rangeMin,rangeMax)

    def add_bias_range(self, affineLayerIndex, neuronIndex, rangeMin, rangeMax):
        trueLayerIndex = None
        cnt = 0
        for (idx,layer) in enumerate(self.layers):
            layer: DP_DNN_layer = layer
            if layer.layer_type == DP_DNN_layer.AFFINE_LAYER:
                cnt += 1
                if cnt == affineLayerIndex:
                    trueLayerIndex = idx
                    break
        if trueLayerIndex == None:
            raise ValueError("The affineLayerIndex is out of range")
        if (trueLayerIndex,neuronIndex) in self.bias_variables:
            raise ValueError("The bias of the neuron has been set before")
        self.bias_variables[(trueLayerIndex,neuronIndex)] = (rangeMin,rangeMax)
        # print("self.bias_variables", self.bias_variables)
        
    def get_intermediate_bounds(self):
        res = []
        affinelayerIndex = 1
        for idx, layer in enumerate(self.layers):
            if layer.layer_type == DP_DNN_layer.AFFINE_LAYER and layer.auxLayer == False:
                    thisLayerConcrete = [ (neuron.historical_concrete_lowest_lower, neuron.historical_concrete_highest_upper) \
                                           for neuron in layer.neurons]
                    assert len(thisLayerConcrete) <= self.layerSizes[affinelayerIndex], \
                            f"{len(thisLayerConcrete)} > {self.layerSizes[affinelayerIndex]}"
                    res.append(thisLayerConcrete[:self.layerSizes[affinelayerIndex]])
                    affinelayerIndex += 1
        return res
    
    def get_post_activation_bounds(self):
        res = []
        for idx, layer in enumerate(self.layers):
            if layer.layer_type in [DP_DNN_layer.RELU_LAYER, DP_DNN_layer.SIGMOID_LAYER, DP_DNN_layer.TANH_LAYER]:
                thisLayerConcrete = [ (neuron.concrete_lower, neuron.concrete_upper) for neuron in layer.neurons]
                res.append(thisLayerConcrete)
        return res



class DP_QNN_network(DP_DNN_network): #only quantifying the weight and the bias. relu is not. signed int with scaling factors(\Delta w)
    def __init__(self, bit_all, DeltaWs, ifSignedOutput):
        super(DP_QNN_network,self).__init__(ifSignedOutput)
        self.bit_all = bit_all
        self.DeltaWs = deepcopy(DeltaWs)        
    
    def create_network(self, weights, bias):
        super().create_network(weights, bias)
        cnt = 0
        for layer in self.layers:
            if layer.layer_type == DP_DNN_layer.AFFINE_LAYER and layer.auxLayer == False:
                assert cnt < len(self.DeltaWs)
                layer.DeltaW = self.DeltaWs[cnt]
                cnt = cnt + 1
        assert cnt == len(self.DeltaWs), f"{cnt} != {len(self.DeltaWs)}"
    
    def load_dnn(self, quantized_model):
        super().load_dnn(quantized_model)
        #assign DealtaWs to all affine layers 
        cnt = 0
        for layer in self.layers:
            if layer.layer_type == DP_DNN_layer.AFFINE_LAYER and layer.auxLayer == False:
                assert cnt < len(self.DeltaWs)
                layer.DeltaW = self.DeltaWs[cnt]
                cnt += 1
        assert cnt == len(self.DeltaWs), f"{cnt} != {len(self.DeltaWs)}"

    def dump(self) -> dict:
        jdict = {}
        jdict["weights"]= []
        jdict["bias"] = []
        jdict["DeltaWs"] = []
        jdict["bit_all"] = self.bit_all
        cnt = 0
        for layer in self.layers:
            layer:DP_DNN_layer = layer
            if layer.layer_type == DP_DNN_layer.AFFINE_LAYER:
                jdict["weights"].append([])
                jdict["bias"].append([])
                jdict["DeltaWs"].append(layer.DeltaW)
                for neuron in layer.neurons:
                    jdict["weights"][cnt].append(neuron.weight.tolist())
                    jdict["bias"][cnt].append(float(neuron.bias))
                cnt += 1
        return jdict