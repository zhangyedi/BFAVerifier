import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import sys
from gurobipy import GRB
import math
from utils.DeepPoly_DNN import *


# C_m^n
def myCombSum(m, n):
    res = 0
    for i in range(n):
        res += math.factorial(m) // (
                math.factorial(i + 1) * math.factorial(m - i - 1))
    return res


def validate_BFA(model, Delta_LL, original_output, counterexample, output, BFA_info, verbose):
    BFA_pos = BFA_info[0]
    BFA_value = BFA_info[1]

    original_weights = model.layers[BFA_pos[0]].get_weights()
    flipped_para = BFA_value * Delta_LL[BFA_pos[0]]
    if BFA_pos[2] == None:  # bias attack
        original_para = original_weights[1][BFA_pos[1]]
        original_weights[1][BFA_pos[1]] = flipped_para
        new_weight = [original_weights[0], original_weights[1]]

    else:  # weight attack
        original_para = original_weights[0][BFA_pos[2]][BFA_pos[1]]
        original_weights[0][BFA_pos[2]][BFA_pos[1]] = flipped_para
        new_weight = [original_weights[0], original_weights[1]]

    print("\nWe find a counterexample where the BFA position is ", BFA_pos)
    print("The original and the flipped integer values are ", np.round(original_para / Delta_LL[BFA_pos[0]]), " and ",
          BFA_value)
    print("The original and the flipped fixed-point values are ", original_para, " and ",
          flipped_para)

    model.layers[BFA_pos[0]].set_weights(new_weight)

    if verbose:
        print("\nThe prediction class (ground-truth) is: ", np.argmax(original_output))
        print("\nThe prediction class (counter-example): ", np.argmax(output))
        print("\nThe output of ground-truth is: ", original_output)
        print("\nThe output of the counter-example: ", output)
        print("\nThe output of counter-example (validation) is: ",
              model.predict(np.expand_dims(counterexample * 255, 0), verbose=False)[0])

    assert (np.argmax(model.predict(np.expand_dims(counterexample * 255, 0), verbose=False)[0]) == np.argmax(output))


class QNNEncoding_MILP:
    def __init__(self, deepModel, W_dict, Delta_LL, args, all_lb_LL, all_ub_LL, verbose=None):
        self.deepModel = deepModel
        self.W_dict = W_dict
        self.Delta_LL = Delta_LL

        self._verbose = verbose
        self.path = args.outputPath
        self.qu_bit = args.qu_bit
        self.flip_bit = args.flip_bit
        self.bfa_vecNum = myCombSum(args.qu_bit, args.flip_bit)
        print("bfa_vecNum: ", self.bfa_vecNum)
        # self.sumOfRealFlip = None
        self.sample_id = args.sample_id
        self.rad = args.rad
        self.all_lb_LL = all_lb_LL
        self.all_ub_LL = all_ub_LL
        self.encoded_dense_layer = []
        self.allBinVar = []
        self.allBinVar_key = []
        # 初始化 Gurobi model
        self.gp_model = gp.Model("qnn_ilp_verifier")
        # self.gp_model.Params.IntFeasTol = 1e-6
        self.gp_model.setParam(GRB.Param.Threads, 120)  # 默认Gurobi的线程数

        # 运行参数
        self._stats = {
            "encoding_time": 0,
            "solving_time": 0,
            "total_time": 0,
            "solving_res": False  # False: Robust; True: find a counter-example, thus Non-Robust
        }

        for i, l in enumerate(deepModel.dense_layers):
            ifLast = False
            if (i == len(deepModel.dense_layers) - 1):
                ifLast = True

            # 为每个 non-input layer 构造 milp 变量
            self.encoded_dense_layer.append(
                LayerEncoding_MILP(
                    layer_size=l.units,
                    gp_model=self.gp_model,
                    qu_bit=self.qu_bit,
                    lb_LL=self.all_lb_LL[i],
                    ub_LL=self.all_ub_LL[i],
                    signed_output=l.signed_output,
                    if_last=ifLast,
                )
            )

        # create input variables for input layer
        input_size = deepModel._input_shape[-1]
        self.input_layer = LayerEncoding_MILP(
            layer_size=input_size,
            gp_model=self.gp_model,
            qu_bit=self.qu_bit,
            if_input=True,
        )

        self.input_gp_vars = self.input_layer.gp_DNN_vars_pos
        self.output_gp_vars = self.encoded_dense_layer[-1].gp_DNN_vars

        self.deepPolyNets_DNN = DP_DNN_network(True)
        self.deepPolyNets_DNN.load_dnn(deepModel)

    def encode(self):

        # encode QNN under BFA
        current_layer = self.input_layer

        for i, l in enumerate(self.encoded_dense_layer):
            tf_layer = self.deepModel.dense_layers[i]

            w_fp, b_fp = tf_layer.get_weights()

            self.encode_dense(current_layer, l, i, w_fp, b_fp)

            current_layer = l

        # encode Sum of Binary Variable is 1

        sumOfBin = np.sum(self.allBinVar)
        self.gp_model.addConstr(sumOfBin <= 1)
        self.gp_model.addConstr(sumOfBin >= 1)

        self.gp_model.update()

    # encode each non-input layer (affine + ReLU) under BFA attacks
    def encode_dense(self, in_layer, out_layer, layer_index, w_fp, b_fp):

        W_dict = self.W_dict

        for out_index in range(out_layer.layer_size):
            weight_row = w_fp[:, out_index]
            bias_fp = b_fp[out_index]

            # encode original affine function
            acc = np.dot(weight_row, in_layer.gp_DNN_vars_pos) + bias_fp

            all_keys = W_dict.keys()

            keyList = list(filter(lambda x: x[0] == layer_index and x[1] == out_index, all_keys))

            # TODO: 优化encoding，用矩阵的方式加速编码
            # encode modified parameter's effect on affine function
            for key in keyList:

                binVarSet = [self.gp_model.addVar(vtype=GRB.BINARY) for i in range(self.bfa_vecNum)]
                self.allBinVar += binVarSet

                self.allBinVar_key.append(key)

                if key[2] == None:  # flip bias
                    valueSet = [(x * self.Delta_LL[layer_index] - bias_fp) for x in W_dict[key]]
                    acc = acc + np.dot(valueSet, binVarSet)
                else:  # flip weight of in_index
                    # self.allBinVar_key.append([key, 'weight'])
                    valueSet = [x * self.Delta_LL[layer_index] - weight_row[key[2]] for x in W_dict[key]]
                    acc = acc + np.dot(valueSet, binVarSet) * in_layer.gp_DNN_vars_pos[key[2]]

            self.gp_model.update()
            self.gp_model.addConstr(acc == out_layer.gp_DNN_vars[out_index])

            # encode ReLU function
            if not out_layer.if_last:  # hidden_layer's ReLU function
                self.gp_model.addGenConstrMax(out_layer.gp_DNN_vars_pos[out_index],
                                              [out_layer.gp_DNN_vars[out_index], 0])
                self.gp_model.update()

    # encoding the problem and solve
    def sat(self, args):
        encode_start_time = time.time()
        self.encode()
        solving_start_time = time.time()

        print(
            "\n==================================== Now we start do ilp-based solving! ====================================")

        numAllVars = self.gp_model.getAttr("NumVars")
        numIntVars = self.gp_model.getAttr("NumIntVars")
        numBinVars = self.gp_model.getAttr("NumBinVars")
        numConstrs = self.gp_model.getAttr("NumConstrs")
        #
        print("The num of vars: ", str(numAllVars))
        print("The num of numIntVars: ", str(numIntVars))
        print("The num of numBinVars: ", str(numBinVars))
        print("The num of Constraints: ", str(numConstrs))

        # self.output_not_argmax_gurobi(prediction)
        # set timeout
        self.gp_model.Params.TimeLimit = 3600
        self.gp_model.Params.NonConvex = 2
        self.gp_model.optimize()
        ifgpUNSat = self.gp_model.status == GRB.INFEASIBLE
        isTimeOut = self.gp_model.status == GRB.TIME_LIMIT

        solving_end_time = time.time()
        self._stats["encoding_time"] = solving_start_time - encode_start_time
        self._stats["solving_time"] = solving_end_time - solving_start_time
        self._stats["total_time"] = solving_end_time - encode_start_time

        print("\nThe total encoding time is: " + str(self._stats["encoding_time"]))
        print("\nThe total solving time is: " + str(self._stats["solving_time"]))
        print("\nThe total time is: " + str(self._stats["total_time"]))
        if isTimeOut:
          print("Timelimit Exceeded!")
          exit(0)
        if ifgpUNSat:
            vadi = "BFA-tolerant Robustness Property is True."
        else:
            vadi = "BFA-tolerant Robustness Property is False."

        print(vadi)

        fo = open(args.outputPath + "/" + str(args.sample_id) + "_attack_" + str(args.rad) + "_qu_bit_" + str(
            args.qu_bit) + "_gp.txt", "w")

        fo.write("Verification Result: " + str(ifgpUNSat) + "\n")
        fo.write(vadi + "\n")
        fo.write("Encoding Time: " + str(self._stats["encoding_time"]) + "\n")
        fo.write("Solving Time: " + str(self._stats["solving_time"]) + "\n")
        fo.write("Total Time: " + str(self._stats["encoding_time"] + self._stats["solving_time"]) + "\n")

        return ifgpUNSat

    def assert_input_box(self, x, rad):

        low_cont, high_cont = np.float32((x - rad) / 255), np.float32((x + rad) / 255)

        input_size = len(self.input_gp_vars)

        # Ensure low_cont is a vector
        low_cont = np.array(low_cont, dtype=np.float32) * np.ones(input_size, dtype=np.float32)
        high_cont = np.array(high_cont, dtype=np.float32) * np.ones(input_size, dtype=np.float32)

        low_cont = np.float32(np.clip(low_cont, 0, 1))
        high_cont = np.float32(np.clip(high_cont, 0, 1))

        for i in range(input_size):
            self.gp_model.addConstr(self.input_layer.gp_DNN_vars_pos[i] <= high_cont[i])
            self.gp_model.addConstr(self.input_layer.gp_DNN_vars_pos[i] >= low_cont[i])

        ################## For checking robustness of original QNN: start ##################
        self.deepPolyNets_DNN.property_region = 1

        for i in range(self.deepPolyNets_DNN.layerSizes[0]):
            self.deepPolyNets_DNN.layers[0].neurons[i].concrete_lower = low_cont[i]
            self.deepPolyNets_DNN.layers[0].neurons[i].concrete_upper = high_cont[i]
            self.deepPolyNets_DNN.property_region *= (high_cont[i] - low_cont[i])
            self.deepPolyNets_DNN.layers[0].neurons[i].concrete_algebra_lower = np.array([low_cont[i]])
            self.deepPolyNets_DNN.layers[0].neurons[i].concrete_algebra_upper = np.array([high_cont[i]])
            self.deepPolyNets_DNN.layers[0].neurons[i].algebra_lower = np.array([low_cont[i]])
            self.deepPolyNets_DNN.layers[0].neurons[i].algebra_upper = np.array([high_cont[i]])
        ################## For checking robustness of original QNN: end ##################
    def assert_input_given(self, low_cont, high_cont):

        # low_cont, high_cont = np.float32((x - rad) / 255), np.float32((x + rad) / 255)

        input_size = len(self.input_gp_vars)

        # Ensure low_cont is a vector
        low_cont = np.array(low_cont, dtype=np.float32) * np.ones(input_size, dtype=np.float32)
        high_cont = np.array(high_cont, dtype=np.float32) * np.ones(input_size, dtype=np.float32)

        # low_cont = np.float32(np.clip(low_cont, 0, 1))
        # high_cont = np.float32(np.clip(high_cont, 0, 1))

        for i in range(input_size):
            self.gp_model.addConstr(self.input_layer.gp_DNN_vars_pos[i] <= high_cont[i])
            self.gp_model.addConstr(self.input_layer.gp_DNN_vars_pos[i] >= low_cont[i])

        ################## For checking robustness of original QNN: start ##################
        # self.deepPolyNets_DNN.property_region = 1

        # for i in range(self.deepPolyNets_DNN.layerSizes[0]):
        #     self.deepPolyNets_DNN.layers[0].neurons[i].concrete_lower = low_cont[i]
        #     self.deepPolyNets_DNN.layers[0].neurons[i].concrete_upper = high_cont[i]
        #     self.deepPolyNets_DNN.property_region *= (high_cont[i] - low_cont[i])
        #     self.deepPolyNets_DNN.layers[0].neurons[i].concrete_algebra_lower = np.array([low_cont[i]])
        #     self.deepPolyNets_DNN.layers[0].neurons[i].concrete_algebra_upper = np.array([high_cont[i]])
        #     self.deepPolyNets_DNN.layers[0].neurons[i].algebra_lower = np.array([low_cont[i]])
        #     self.deepPolyNets_DNN.layers[0].neurons[i].algebra_upper = np.array([high_cont[i]])
        ################## For checking robustness of original QNN: end ##################

    def output_not_argmax_gurobi(self, max_index):
        # bigM = GRB.MAXINT
        bigM = 1000

        k_list = []
        for i in range(len(self.output_gp_vars)):
            # for i in range(10):
            if i == int(max_index):
                continue

            k_i = self.gp_model.addVar(vtype=GRB.BINARY)
            k_list.append(k_i)
            self.gp_model.update()
            self.gp_model.addConstr(
                self.output_gp_vars[i] >= self.output_gp_vars[int(max_index)] + bigM * (k_i - 1))
            self.gp_model.addConstr(
                self.output_gp_vars[i] <= self.output_gp_vars[int(max_index)] + bigM * k_i - 1)

        sum_constr = 0
        self.gp_model.update()

        for i in range(len(k_list)):
            sum_constr = sum_constr + k_list[i]

        self.gp_model.addConstr(sum_constr >= 1)

        # self.gp_model.update()

    def get_input_assignment(self):
        input_values = np.array(
            [var.X for var in self.input_layer.gp_DNN_vars_pos]
        )

        output_values = np.array(
            [var.X for var in self.encoded_dense_layer[-1].gp_DNN_vars]
        )

        # sumOfRealFlip = self.sumOfRealFlip.X

        # Todo: 定位vulnerable parameter's value
        # weight_BFA_values = np.array()
        binaryVars = np.array(
            [var.X for var in self.allBinVar], dtype=np.float32
        )

        # [BFA_pos, BFA_value]
        BFA_info = self.compute_attack_vector(binaryVars)

        return np.array(input_values, dtype=np.float32), np.array(output_values, dtype=np.float32), BFA_info

    # compute back the attack vectors
    def compute_attack_vector(self, binaryVars):

        # 以防binaryVars返回0.99999999
        assert np.round(np.sum(binaryVars)) == 1
        binaryVars = [np.round(ele) for ele in binaryVars]

        pos = list(binaryVars).index(1)

        key_index = pos // self.bfa_vecNum
        item_index = pos % self.bfa_vecNum

        key = self.allBinVar_key[key_index]

        # 返回对应的weight的位置，以及翻转后的value
        return [key, self.W_dict[key][item_index]]


class LayerEncoding_MILP:
    def __init__(self, layer_size, gp_model, qu_bit, lb_LL=None, ub_LL=None, signed_output=False, if_last=False,
                 if_input=False):
        self.layer_size = layer_size
        self.qu_bit = qu_bit
        self.signed_output = signed_output
        self.if_last = if_last
        self.if_input = if_input

        # TODO: 这里应通过DeepPolyR获取到相应的neuron的上下界
        if signed_output:
            self.gp_DNN_vars = [
                gp_model.addVar(lb=lb_LL[i], ub=ub_LL[i], vtype=GRB.CONTINUOUS) for i in range(layer_size)  # [-100,100]
            ]
        else:
            if self.if_input:
                self.gp_DNN_vars_pos = [
                    gp_model.addVar(lb=-1, ub=1, vtype=GRB.CONTINUOUS) for i in range(layer_size)  # [-100,100]
                ]
            else:
                # hidden layer: add vars before relu
                self.gp_DNN_vars = [
                    gp_model.addVar(lb=lb_LL[i], ub=ub_LL[i], vtype=GRB.CONTINUOUS) for i in range(layer_size)
                ]
                self.gp_DNN_vars_pos = [
                    gp_model.addVar(ub=ub_LL[i], vtype=GRB.CONTINUOUS) for i in range(layer_size)
                ]

        gp_model.update()


def check_robustness_gurobi(milp_encoding, args, prediction, x_lb, x_ub):
    # x = x.flatten()

    # Step1: encode input region I^r_u
    # milp_encoding.assert_input_box(x, args.rad)
    
    milp_encoding.assert_input_given(x_lb, x_ub)

    ####### 以下内容为 Cbeck Robustness Start: if not robust for original QNN, then exit
    # milp_encoding.deepPolyNets_DNN.add_difference_layer(prediction)
    # milp_encoding.deepPolyNets_DNN.deeppoly()

    # for out_index in range(len(milp_encoding.deepPolyNets_DNN.layers[-1].neurons)):
    #     if out_index != prediction:
    #         if milp_encoding.deepPolyNets_DNN.layers[-1].neurons[out_index].concrete_lower < 0:
    #             print("############ The robustness property is not hold in the original DNN. Hence we exit out?")
    #             # exit(0)

    # print("The robustness hold!")
            

    ####### Cbeck Robustness End: if not robust for original QNN, then exit

    # Step2: MILP encode output property
    milp_encoding.output_not_argmax_gurobi(int(prediction))

    # Step3: MILP encode QNN under BFAs w.r.t vulnerable paraeter set W
    ifUNSat = milp_encoding.sat(args)

    if not ifUNSat:  #
        attack, output, BFA_info = milp_encoding.get_input_assignment()
        return (False, attack, output, BFA_info)
    else:
        return (True, None, None, None)
