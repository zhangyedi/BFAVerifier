#include <vector>
#include <iostream>
#include <fstream>
#include "json.hpp"
#include "gpupoly.h"
#include "bitflip_utils.hpp"
#include "dpolyr.h"

std::vector<target> getTargets(int layerIndex, int layerSize, int lastLayerSize, int targets_num){
    std::vector<target> res;
    if(targets_num == -1)
        targets_num = layerSize * lastLayerSize;
    assert(targets_num <= layerSize * lastLayerSize);
    auto convert = [&](int numerized) -> target{
        return {layerIndex, numerized / lastLayerSize, numerized % lastLayerSize};
    };
    std::vector<int> numerized_targets;
    numerized_targets.resize(targets_num);
    for(int i = 0; i < targets_num; i++){
        numerized_targets[i] = i; //first be like this
    }
    // random_shuffle(numerized_targets.begin(), numerized_targets.end());

    numerized_targets.resize(targets_num);
    for(int i = 0; i < targets_num; i++){
        res.push_back(convert(numerized_targets[i]));
    }
    return res;
}


int add_variable_bias(NeuralNetwork *nn, int layerIndex, int neuronIndex, double rangeMin, double rangeMax){
    return changeBiasSingle(nn, layerIndex, neuronIndex, rangeMin, rangeMax);
}

int add_variable_weight(NeuralNetwork *nn, int layerIndex, int neuronIndex, int weightIndex, double rangeMin, double rangeMax, int num_virtual_neuron){
    int reluLayerIndex = layerIndex - 1;
    int reluLayerInfoSize = getReLUInfoSize(nn, reluLayerIndex);
    int reluLayerSize = getOutputSize(nn, reluLayerIndex);
    int linearLayerSize = getOutputSize(nn, layerIndex);
    if(linearLayerSize == -1){
        std::cout << "Error, not a linear layer" << std::endl;
        return -1;
    }
    if(layerIndex != 1 && reluLayerInfoSize == -1){
        std::cout << "Error, previous layer not a ReLU layer" << std::endl;
        return -1;
    }

    if(layerIndex == 1){
        return -1;
    }
    else{
        // std::cout << "layerIndex" << layerIndex << "reluLayerInfoSize" << reluLayerInfoSize << "reluLayerSize" << reluLayerSize << "linearLayerSize" << linearLayerSize << std::endl;
        int virtualNeuronId = reluLayerSize - num_virtual_neuron + reluLayerInfoSize;
        if(virtualNeuronId >= reluLayerSize){
            std::cout << "Error, not enough virtual neurons" << std::endl;
            return -1;
        }
        DPolyRInfo info = {
            .attackType = DPolyRInfo::AttackType::BITFLIP_RANGE_ABSTRACT,
            .rangeMin = rangeMin,
            .rangeMax = rangeMax,
            .preNeuronIndex = weightIndex,
            .curVirtualNeuronIndex = virtualNeuronId,
        };
        // info.print();
        int res1 = changeLinear_single_d(nn, layerIndex, neuronIndex, weightIndex, 0);
        int res2 = changeLinear_single_d(nn, layerIndex, neuronIndex, virtualNeuronId, 1);
        // printf("changed %d %d %d to %d\n", layerIndex, neuronIndex, weightIndex, 0);
        // printf("changed %d %d %d to %d\n", layerIndex, neuronIndex, virtualNeuronId, 1);
        if(res1 == -1 || res2 == -1){
            std::cout << "Error changing weight" << std::endl;
            return -1;
        }
        if(addReLUInfo(nn, reluLayerIndex, &info) != 0){
            std::cout << "Error adding ReLU info" << std::endl;
            return -1;
        }
    }
    return 0;
}

NeuralNetwork * createNN(const std::vector<int>& layerSizes,const std::vector<std::vector<std::vector<double>>>&weights, const std::vector<std::vector<double>>& bias, int virtualNeuronNum, size_t& trueLayerSize){
    NeuralNetwork *nn = create(layerSizes[0] + virtualNeuronNum);
    if(!nn){
        std::cout << "Error creating network" << std::endl;
        return nullptr;
    }
    //check layer shapes
    for(int i = 1; i < layerSizes.size(); i++){
        int shapeRows = weights[i].size();
        int shapeCols = weights[i][0].size();
        if(shapeCols != layerSizes[i - 1]){
            std::cout << "Error, weight shape does not match: " << shapeCols << ' '<< layerSizes[i-1] << std::endl;
            return nullptr;
        }
        if(shapeRows != layerSizes[i]){
            std::cout << "Error, weight shape does not match" << std::endl;
            return nullptr;
        }
    }
    const int MAXN = 1 << 22; //max size 2048
    double * temp = new double[MAXN];
    int prev = 0;
    for(int i = 1; i < layerSizes.size(); i++){
        int shapeRows = weights[i].size();
        int shapeCols = weights[i][0].size();
        int vshapeCols = shapeCols + virtualNeuronNum;
        assert(vshapeCols * shapeRows <= MAXN);
        for(int j = 0; j < shapeRows; j++){
            memcpy(temp + j * vshapeCols, weights[i][j].data(), shapeCols * sizeof(double));
            memset(temp + j * vshapeCols + shapeCols, 0, virtualNeuronNum * sizeof(double));
        }
        prev = addLinear_d(nn, prev, shapeRows, temp);
        if(bias.size() && bias[i].size()){
            if(bias[i].size() != shapeRows){
                std::cout << "Error, bias shape does not match" << std::endl;
                delete [] temp;
                return nullptr;
            }
            assert(bias[i].size() == shapeRows);
            prev = addBias_d(nn, prev, bias[i].data());
        }
        if(i != layerSizes.size() - 1)
            prev = addReLUVirtual(nn, shapeRows + virtualNeuronNum, prev, true);
    }
    trueLayerSize = prev;
    delete [] temp;
    return nn;
}


//once added, can only restore the network by reset info. must contain all infos!!

int resetInfo(NeuralNetwork *nn, std::vector<int> layerIndexes, std::vector<int> neuronIndexes, std::vector<int> weightIndexes, std::vector<double> originVals, int num_virtual_bit){
    for(int i = 0; i < layerIndexes.size(); i++){
        int layerIndex = layerIndexes[i];
        int neuronIndex = neuronIndexes[i];
        int weightIndex = weightIndexes[i];
        if(weightIndex == -1){
            //clear a relu
            int res = changeBiasSingle(nn, layerIndex, neuronIndex, originVals[i], originVals[i]);
            if (res != 0){
                std::cout << "Error restoring bias" << std::endl;
                return -1;
            }
            continue;
        }
        double val = originVals[i];
        if(neuronIndex < 0 || neuronIndex >= getOutputSize(nn, layerIndex)){
            std::cout << "Error, neuron index out of range" << std::endl;
            return -1;
        }
        if(weightIndex < 0 || weightIndex >= getOutputSize(nn, layerIndex - 1)){
            std::cout << "Error, weight index out of range" << std::endl;
            return -1;
        }
        if(layerIndex != 1 && clearReLUInfo(nn, layerIndex - 1) != 0){
            std::cout << "Error clearing ReLU info" << std::endl;
            return -1;
        }
        if(layerIndex == 1){
            return -1;
        }
        // printf("restored %d %d %d to %f\n", layerIndex, neuronIndex, weightIndex, val);
        changeLinear_single_d(nn, layerIndex, neuronIndex, weightIndex, val);
        assert(getOutputSize(nn, layerIndex - 1) - num_virtual_bit >= 0);
        for(int vi = getOutputSize(nn, layerIndex - 1) - num_virtual_bit; vi < getOutputSize(nn, layerIndex - 1); vi++){
            // printf("resetting %d %d %d\n", layerIndex, neuronIndex, vi);
            changeLinear_single_d(nn, layerIndex, neuronIndex, vi, 0);
        }
    }
    return 0;
}

int getTrueAffineIndex(NeuralNetwork *nn, int affineIndex, size_t trueLayerSize){
    if(affineIndex < 0){
        return -1;
    }
    size_t cnt = 0;
    for(size_t i = 0; i < trueLayerSize; ++i){
        if(getLinearRowSize(nn, i) > 0){
            cnt += 1;
            if(cnt == affineIndex){
                return i;
            }
        }
    }
    return -1;
}

// Flip cnt bits of the value except for the sign bit. Use bit_all-bit two's complement form. return the correct flipped value. return the value's range
std::pair<int, int> rangeFlipKBitIntPreserve(int value, int bit_all, int cnt, int mask)
{
        if (cnt == 0)
        {
                int temp = flip_bit_mask(value, mask, bit_all);
                return std::make_pair(temp, temp);
        }
        std::pair<int, int> res = std::make_pair(10000000, -10000000);
        for (int i = 0; i < bit_all - 1; i++)
        {
                if ((mask >> i) & 1)
                {
                        continue;
                }
                std::pair<int, int> temp = rangeFlipKBitIntPreserve(value, bit_all, cnt - 1, mask | (1 << i));
                res.first = std::min(res.first, temp.first);
                res.second = std::max(res.second, temp.second);
        }
        return res;
}

// Flip cnt bits of the value. Use bit_all-bit two's complement form. return the correct flipped value. return the value's range
std::pair<int, int> rangeFlipKBitInt(int value, int bit_all, int cnt, int mask)
{
        if (cnt == 0)
        {
                int temp = flip_bit_mask(value, mask, bit_all);
                return std::make_pair(temp, temp);
        }
        std::pair<int, int> res = std::make_pair(10000000, -10000000);
        for (int i = 0; i < bit_all; i++)
        {
                if ((mask >> i) & 1)
                {
                        continue;
                }
                std::pair<int, int> temp = rangeFlipKBitInt(value, bit_all, cnt - 1, mask | (1 << i));
                res.first = std::min(res.first, temp.first);
                res.second = std::max(res.second, temp.second);
        }
        return res;
}

// Flip cnt bits of the value. Use bit_all-bit two's complement form. return the correct flipped value. return the value's range
std::pair<int, int> rangeFlipKBitIntMasks(int value, int bit_all, const std::vector<int>& masks)
{
        if(masks.size() == 0)       
                return std::make_pair(value, value);
        
        std::pair<int, int> res = std::make_pair(10000000, -10000000);
        for (int mask: masks)
        {
                int temp = flip_bit_mask(value, mask, bit_all);
                res.first = std::min(res.first, temp);
                res.second = std::max(res.second, temp);
        }
        return res;
}


// Flip the masked bit of the value. Use bit_all-bit two's complement form. return the correct flipped value
int flip_bit_mask(int value, int mask, int bit_all)
{
        assert(value <= (1 << (bit_all - 1)) - 1);
        assert(value >= -(1 << (bit_all - 1)));
        int mask0 = (1 << bit_all) - 1;
        value &= mask0;
        value ^= mask;
        if (value >> (bit_all - 1))
        {
                value |= (0xffffffff << bit_all);
        }
        return value;
}

std::vector<int> getBitMask(int bit_all, int cnt){
    std::vector<int> res;
    assert(bit_all <= 31);
    long upper = 1 << bit_all;
    for(int S = 0; S < upper; ++S){
        if(__builtin_popcount(S) <= cnt){
            res.push_back(S);
        }
    }
    return res;
}

// Flip the k-th bit of the value. Use bit_all-bit two's complement form. return the correct flipped value
int flip_bit(int value, int k_th, int bit_all)
{
        assert(k_th >= 0 && k_th < bit_all);
        assert(value <= (1 << (bit_all - 1)) - 1);
        assert(value >= -(1 << (bit_all - 1)));
        int mask = (1 << bit_all) - 1;
        value &= mask;
        value ^= (1 << k_th);
        if (value >> (bit_all - 1))
        {
                value |= (0xffffffff << bit_all);
        }
        return value;
}


int readJson(std::string path, nnInfo& info){
    std::ifstream in(path);
    if(!in){
        std::cout << "Error opening file" << std::endl;
        return -1;
    }
    nlohmann::json jdata = nlohmann::json::parse(in);
    //weights,bias,DeltaWs,bit_all,input_lower,input_upper,label
    //weights are [][][] double
    //bias are [][] double
    //DeltaWs are [] double
    std::string weight_path_string = jdata.find("weight_path") == jdata.end() ? "(not given) " + path : jdata["weight_path"].get<std::string>();
    std::cout << "weights_path: " << weight_path_string << std::endl;
    info = {};
    info.weights = jdata["weights"].get<std::vector<std::vector<std::vector<double>>>>();
    info.bias = jdata["bias"].get<std::vector<std::vector<double>>>();
    info.DeltaWs = jdata["DeltaWs"].get<std::vector<double>>();
    info.input_lower = jdata["input_lower"].get<std::vector<double>>();
    info.input_upper = jdata["input_upper"].get<std::vector<double>>();
    info.label = jdata["label"].get<int>();
    info.bit_all = jdata["bit_all"].get<int>();
    info.layerSizes = {};
    for (auto& w: info.weights){
        info.layerSizes.push_back(w.size());
    }
    //check data validity. 
    if(info.weights.size() != info.bias.size()){
        std::cout << "Error, weights and bias size mismatch" << std::endl;
        return -1;
    }
    if(info.input_lower.size() != info.input_upper.size()){
        std::cout << "Error, input_lower and input_upper size mismatch" << std::endl;
        return -1;
    }
    if(info.DeltaWs.size() != info.weights.size()){
        std::cout << "Error, DeltaWs size mismatch" << std::endl;
        return -1;
    }
    for(int i = 0; i < info.weights.size(); i++){
        if(info.weights[i].size() != info.bias[i].size()){
            std::cout << "Error, weights and bias size mismatch" << std::endl;
            return -1;
        }
        size_t lastLayerSize = i == 0 ? info.input_lower.size() : info.weights[i - 1].size();
        if(info.weights[i][0].size() != lastLayerSize){
            std::cout << "Error, weight size mismatch" << std::endl;
            return -1;
        }
        for(auto& w: info.weights[i]){
            if(w.size() != info.weights[i][0].size()){
                std::cout << "Error, weight size mismatch" << std::endl;
                return -1;
            }
        }
    }

    std::cout<<"====\nAppending input layer\n";
    std::cout << "info.layerSizes.size() = "<<info.layerSizes.size()<<std::endl; 
    info.layerSizes.insert(info.layerSizes.begin(), info.input_lower.size());
    std::cout << "info.layerSizes.size() = "<<info.layerSizes.size()<<std::endl;
    std::cout << "info.weights.size() = "<<info.weights.size()<<std::endl;
    info.weights.insert(info.weights.begin(), std::vector<std::vector<double>>(1,std::vector<double>(0)));
    std::cout << "info.weights.size() = "<<info.weights.size()<<std::endl;
    std::cout << "info.bias.size() = "<<info.bias.size()<<std::endl;
    info.bias.insert(info.bias.begin(), std::vector<double>(1));
    std::cout << "info.bias.size() = "<<info.bias.size()<<std::endl;
    info.DeltaWs.insert(info.DeltaWs.begin(), 0);

    std::cout<<"weights.size()'s are ";
    for(auto& w: info.weights)
        std::cout<<"("<<w.size()<<','<< w[0].size()<<") ";
    std::cout <<std::endl;
    std::cout<<"bias.size()'s are ";
    for(auto& b: info.bias)
        std::cout<<b.size()<<' ';
    std::cout <<std::endl;
    std::cout<<"DeltaWs.size() is "<<info.DeltaWs.size()<<std::endl;
    std::cout<<"input_lower.size() is "<<info.input_lower.size()<<std::endl;
    std::cout<<"input_upper.size() is "<<info.input_upper.size()<<std::endl;
    std::cout<<"input_id is "<<info.input_id<<std::endl;
    std::cout<<"label is "<<info.label<<std::endl;
    std::cout<<"bit_all is "<<info.bit_all<<std::endl;
    std::cout<<"layerSizes are" << std::endl;
    for(auto& l: info.layerSizes){
        std::cout << l << ' ';
    }
    std::cout<<'\n';

    return 0;
}
