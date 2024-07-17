#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <bitset>
#include <omp.h>
#include <chrono>
#include "gpupoly.h"
#include "dpolyr.h"
#include "bitflip_utils.hpp"

int validate(){
    std::vector<int> layerSizes = {2, 2, 2, 2};
    int virtualNeuronNum = 2;
    NeuralNetwork *nn[10];
    size_t trueLayerSize = 0;
    std::vector<std::vector<std::vector<double>>> weights = {
        {},
        {{1, 1}, {1, -1}},
        {{1, 1}, {1, -1}},
        {{1, 1}, {1, -1}}
    };
    std::vector<std::vector<double>> bias = {
        {},
        {0, 0},
        {0, 0},
        {0, 0}
    };
    std::vector<int> layerIndexes =     {1,2,3,1,2,3,1,2,3,1};
    std::vector<int> neuronIndexes =    {0,0,0,0,0,1,1,1,1,1};
    std::vector<int> weightIndexes =    {0,1,0,1,0,1,0,1,0,1};
    std::vector<double> rangeMins =     {-5,-4,-3,-2,-1,0,1,2,3,4};
    std::vector<double> rangeMaxs =     {-4,-3,-2,-1,0,1,2,3,4,5};
    std::vector<double> originVals =    {1,1,1,1,1,-1,0,-1,0,-1};
    std::vector<double> lower = {-3,4};
    std::vector<double> upper = {-1,5};
    
    for(int i = 0; i < 10; i++){
        nn[i] = createNN(layerSizes, weights,bias,virtualNeuronNum,trueLayerSize);
        if(!nn[i]){
            std::cout << "Error creating network" << std::endl;
            return -1;
        }
    }
    int T = 0,failures = 0;
    // #pragma omp parallel for
    for(int i = 0; i < 10; ++ i){
        if(layerIndexes[i] == 1)
            continue;
        layerIndexes[i] = getTrueAffineIndex(nn[i], layerIndexes[i], trueLayerSize);
        if(add_variable_weight(nn[i], layerIndexes[i], neuronIndexes[i], weightIndexes[i], rangeMins[i], rangeMaxs[i], virtualNeuronNum) != 0){
            std::cout << "Error adding variable weight" << std::endl;
            failures = 1;
        }
        test_d(nn[i], lower.data(), upper.data(), 0, true);
        resetInfo(nn[i], {layerIndexes[i]}, {neuronIndexes[i]}, {weightIndexes[i]}, {originVals[i]}, virtualNeuronNum);
        ++T;
    }
    std::cout << "Validating Bias" << std::endl;
    for(int i = 0; i < 10; ++i){
        if(layerIndexes[i] == 1)
            continue;
        int biasLayer = layerIndexes[i] + 1;
        if(add_variable_bias(nn[i], biasLayer, neuronIndexes[i], rangeMins[i], rangeMaxs[i]) != 0){
            std::cout << "Error adding variable bias" << std::endl;
            failures = 1;
        }
        test_d(nn[i], lower.data(), upper.data(), 0, true);
        resetInfo(nn[i], {biasLayer}, {neuronIndexes[i]}, {-1}, {0}, virtualNeuronNum);
        ++T;
    }
    for(int i = 0; i < 10; ++ i){
        clean(nn[i]);
    }
    if(failures){
        return -1;
    }
    // std::cout << T << std::endl;
    return 0;
}

enum Method{
    BASELINE, BINARYSEARCH
};

int add_variable_para(NeuralNetwork *nn, 
                    int layerIndex, 
                    int neuronIndex, 
                    int weightIndex, 
                    double rangeMin, 
                    double rangeMax,
                    int virtualNeuronNum){
    if(weightIndex == -1){
        return add_variable_bias(nn, layerIndex, neuronIndex, rangeMin, rangeMax);
    }
    assert(layerIndex != 1);
    return add_variable_weight(nn, layerIndex, neuronIndex, weightIndex, rangeMin, rangeMax, virtualNeuronNum);
}

int changeParaSingle(NeuralNetwork *nn, 
                    int layerIndex, 
                    int neuronIndex, 
                    int weightIndex, 
                    double newValue){
    if(weightIndex == -1){
        return changeBiasSingle(nn, layerIndex, neuronIndex, newValue, newValue);
    }
    return changeLinear_single_d(nn, layerIndex, neuronIndex, weightIndex, newValue);
}

//Return 1 if Proved for all sample with [l,r]
int binaryVerify(NeuralNetwork* nn, 
                    const nnInfo&info, 
                    int layerIndex, 
                    int trueIndex, 
                    int neuronIndex, 
                    int weightIndex, 
                    int virtualNeuronNum, 
                    const std::vector<double> filipped_sample, 
                    int l,int r, int& queries, int degrade){
    if(l > r)
        return 1;
    ++queries;
    static const char* MESSAGE[]={"Fail to prove","Proved"};
    const double original_value = weightIndex == -1 ? info.bias[layerIndex][neuronIndex] 
                                : info.weights[layerIndex][neuronIndex][weightIndex];
    if(l == r){
        changeParaSingle(nn, trueIndex, neuronIndex, weightIndex, filipped_sample[l]);
        int res = test_d(nn, info.input_lower.data(), info.input_upper.data(), info.label, true);
        changeParaSingle(nn, trueIndex, neuronIndex, weightIndex, original_value);
        // printf("(Divide and Counter) %s %d %d %d by DeepPoly at [%d]=%lf\n",MESSAGE[res], layerIndex, neuronIndex, weightIndex, l, filipped_sample[l]);
        return res;
    }
    int resAll = 0;
    if(degrade != 1){
        add_variable_para(nn, trueIndex, neuronIndex, weightIndex, filipped_sample[l], filipped_sample[r], virtualNeuronNum);
        resAll = test_d(nn, info.input_lower.data(), info.input_upper.data(), info.label, true);
        resetInfo(nn, {trueIndex}, {neuronIndex}, {weightIndex}, {original_value}, virtualNeuronNum);
    }
    double valueMid = (filipped_sample[l] + filipped_sample[r]) / 2;
    int mid = std::upper_bound(filipped_sample.begin(), filipped_sample.end(), valueMid) - filipped_sample.begin() - 1;
    if(resAll){
        return 1;
    }
    return binaryVerify(nn, info, layerIndex, trueIndex, neuronIndex, weightIndex, virtualNeuronNum, filipped_sample, l, mid, queries, degrade)
            && binaryVerify(nn, info, layerIndex, trueIndex, neuronIndex, weightIndex, virtualNeuronNum, filipped_sample, mid + 1, r, queries, degrade);
}

#define OMP_NUM_THREADS 1
int verifyDeepPolyR(const nnInfo &info, int targets_per_layer, int bit_flip_cnt, int &queries, Method method){
    int all_proved_neg = 0;
    NeuralNetwork *nn[OMP_NUM_THREADS];
    size_t sizes[OMP_NUM_THREADS];
    int virtualNeuronNum = bit_flip_cnt;
    for(int i = 0; i < OMP_NUM_THREADS; i++){
        nn[i] = createNN(info.layerSizes, info.weights, info.bias, virtualNeuronNum, sizes[i]);
        if(!nn[i]){
            std::cout << "Error creating network" << std::endl;
            return -1;
        }
    }
    std::vector<target> targets;
    std::vector<int> masks = getBitMask(info.bit_all, bit_flip_cnt);
    printf("!DEBUG: Infer once with no flips\n");
    int no_flip_proved = test_d(nn[0], info.input_lower.data(), info.input_upper.data(), info.label, true);
    if(!no_flip_proved){
        std::cout << "Fail to prove with no flips" << std::endl;
        for(int i = 0; i < OMP_NUM_THREADS; i++){
            clean(nn[i]);
        }
        return 1;
    }
    printf("!DEBUG: Infer once with no flips. Proved\n");
    printf("!DEBUG: LOG OPTION: OMIT PROVED PARA\n");
    const int OMIT_PROVED = 1;
    for(int layerIndex = 1; layerIndex < (int)info.weights.size(); ++layerIndex){
        //add weights
        auto thisL = getTargets(layerIndex, info.weights[layerIndex].size(), info.weights[layerIndex][0].size(), targets_per_layer);
        //add bias
        auto thisB = getTargets(layerIndex, info.weights[layerIndex].size(), 1, targets_per_layer);
        for(auto& tar:thisB) tar.weightIndex = -1;
        targets.insert(targets.end(), thisL.begin(), thisL.end());
        targets.insert(targets.end(), thisB.begin(), thisB.end());
    }
    for(int i = 0; i < (int)targets.size(); ++i){
        int tid = omp_get_thread_num();
        int layerIndex = targets[i].layerIndex;
        int weightIndex = targets[i].weightIndex;
        int trueIndex = 
            weightIndex == -1?
            getTrueAffineIndex(nn[tid], layerIndex, sizes[tid]) + 1:
            getTrueAffineIndex(nn[tid], layerIndex, sizes[tid]);
        int neuronIndex = targets[i].neuronIndex;
        const int fixed_value = std::round(info.weights[layerIndex][neuronIndex][weightIndex]
                                / info.DeltaWs[layerIndex]);
        std::vector<double> flipped_samples;
        flipped_samples.reserve(masks.size());
        for(auto mask:masks){
            double value = flip_bit_mask(fixed_value, mask, info.bit_all)
                            * info.DeltaWs[layerIndex];
            flipped_samples.push_back(value);
        }
        sort(flipped_samples.begin(), flipped_samples.end());
        int degrade = (layerIndex == 1) && (weightIndex != -1);
        int res = 0;
        if(degrade){
            double RangeMin = flipped_samples[0];
            double RangeMax = flipped_samples.back();
            double input_rangeMin = info.input_lower[weightIndex];
            double input_rangeMax = info.input_upper[weightIndex];
            double biasValue = info.bias[layerIndex][neuronIndex];
            double weightValue = info.weights[layerIndex][neuronIndex][weightIndex];
            changeParaSingle(nn[tid], trueIndex, neuronIndex, weightIndex, 0);
            double RangeMinProduct = std::min({RangeMin * input_rangeMin, RangeMin * input_rangeMax, RangeMax * input_rangeMin, RangeMax * input_rangeMax});
            double RangeMaxProduct = std::max({RangeMin * input_rangeMin, RangeMin * input_rangeMax, RangeMax * input_rangeMin, RangeMax * input_rangeMax});
            add_variable_para(nn[tid], trueIndex + 1, neuronIndex, -1, RangeMinProduct + biasValue, RangeMaxProduct + biasValue, virtualNeuronNum);
            ++queries;
            res = test_d(nn[tid], info.input_lower.data(), info.input_upper.data(), info.label, true);
            resetInfo(nn[tid], {trueIndex + 1}, {neuronIndex}, {-1}, {biasValue}, virtualNeuronNum);
            changeParaSingle(nn[tid], trueIndex, neuronIndex, weightIndex, weightValue);
        }
        if(res == 0)
            res = binaryVerify(nn[tid], info, layerIndex, trueIndex,
                                neuronIndex, weightIndex, virtualNeuronNum, flipped_samples, 
                                0, (int)flipped_samples.size() - 1, queries, degrade);
        if(res){
            if (!OMIT_PROVED){
                if(weightIndex != -1)
                    printf("(Overall) Proved %d %d %d with all masks. Summary: %d %d %d \n", layerIndex, neuronIndex, weightIndex, res, -1, -1 );
                else
                    printf("(Overall) Proved %d %d (bias) with all masks. Summary: %d %d %d \n", layerIndex, neuronIndex, res, -1, -1);
            }
        }else{
            if(weightIndex != -1)
                printf("(Overall) Fail to prove %d %d %d with all masks. Summary: %d %d %d \n", layerIndex, neuronIndex, weightIndex, res, -1, -1);
            else
                printf("(Overall) Fail to prove %d %d (bias) with all masks. Summary: %d %d %d \n", layerIndex, neuronIndex, res, -1, -1);
            all_proved_neg = 1;
        }
        const int PRINT_INTERVAL = 10000;
        if(i % PRINT_INTERVAL == 0){
            //flush stdout
            // std::cout << "Currently processed " << i << " targets, at layer " 
            // << layerIndex << " with neuron " << neuronIndex << " and weight " 
            // << weightIndex << std::endl;
            printf("Currently processed %d targets in %d queries, at layer %d with neuron %d and weight %d [PRINT_INTERVAL=%d]...\n"
                    , i + 1, queries, layerIndex, neuronIndex, weightIndex,PRINT_INTERVAL);
        }
    }
    for(int i = 0; i < OMP_NUM_THREADS; i++){
        clean(nn[i]);
    }
    return all_proved_neg;
}

#define OMP_NUM_THREADS 1

int main(int argc, char *argv[]){    
    srand(123);
    //set precision of cout
    validate();
    printf("Finished Validating\n-------------\n");

    if(argc < 5){
        std::cout << "Usage: " << argv[0] << " <json file> <method (binarysearch_all)> <targets_per_layer> <bit_flip_cnt>" << std::endl;
        return 1;
    }

    int res = validate();
    if(res != 0){
        std::cout << "Validation failed. Please check your environment!" << std::endl;
        return 2;
    }

    nnInfo info;
    if(readJson(argv[1], info) != 0){
        std::cout << "Error reading json" << std::endl;
        return 1;
    }
    assert(std::string(argv[2]) == "binarysearch_all");
    int targets_per_layer = std::stoi(argv[3]);
    int bit_flip_cnt = std::stoi(argv[4]);
    int queries = 0;

    printf("json file: %s\n",argv[1]);
    printf("method: %s\n",argv[2]);
    printf("targets_per_layer: %d\n",targets_per_layer);
    printf("bit_flip_cnt: %d\n",bit_flip_cnt);

    auto start_time = std::chrono::system_clock::now();
    int res2 = verifyDeepPolyR(info, targets_per_layer, bit_flip_cnt, queries, BINARYSEARCH);
    auto end_time = std::chrono::system_clock::now();
    std::cout << "Elapsed Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1e3 << std::endl;
    std::cout << "flg =" << res2 << std::endl;
    std::cout << "all_proved =" << !res2 << std::endl;
    std::cout << "queries = " << queries << std::endl;
    return 0;
}