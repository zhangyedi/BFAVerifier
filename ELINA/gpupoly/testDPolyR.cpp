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



#define OMP_NUM_THREADS 1
int verifyBaselineWeight(const nnInfo&info, int targets_per_layer, int bit_flip_cnt, int& queries) {
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
    //Warning bit-flip the first layer is not yet supported.
    for(int layerIndex = 2; layerIndex < (int)info.weights.size(); ++layerIndex){
        auto thisL = getTargets(layerIndex, info.weights[layerIndex].size(), info.weights[layerIndex][0].size(), targets_per_layer);
        targets.insert(targets.end(), thisL.begin(), thisL.end());
    }
    std::vector<int> masks = getBitMask(info.bit_all, bit_flip_cnt);
    int all_proved_neg = 0;

    std::cout << "preprocessing end\n"; 
// #pragma omp parallel for num_threads(OMP_NUM_THREADS) 
    for(int i = 0; i < (int)targets.size(); ++i){
        int tid = omp_get_thread_num();
        int layerIndex = targets[i].layerIndex;
        int trueIndex = getTrueAffineIndex(nn[tid], layerIndex, sizes[tid]);
        int neuronIndex = targets[i].neuronIndex;
        int weightIndex = targets[i].weightIndex;
        int local_proved = 1;
        for(auto mask : masks){
            const int fixed_value = std::round(info.weights[layerIndex][neuronIndex][weightIndex] / info.DeltaWs[layerIndex]);
            double value = flip_bit_mask(fixed_value, mask, info.bit_all) * info.DeltaWs[layerIndex];
            changeLinear_single_d(nn[tid], trueIndex, neuronIndex, weightIndex, value);   
            queries += 1;
            if(!test_d(nn[tid], info.input_lower.data(), info.input_upper.data(), info.label, true)){
                all_proved_neg = 1;
                local_proved = 0;
                printf("Fail to prove %d %d %d with mask %s, val=%lf\n", layerIndex, neuronIndex, weightIndex, std::bitset<8>(mask).to_string().c_str(), value);
                break;
            }else
                printf("Proved %d %d %d with mask %s, val=%lf\n", layerIndex, neuronIndex, weightIndex, std::bitset<8>(mask).to_string().c_str(),value);
        }
        if(local_proved)
            printf("Proved %d %d %d with all masks (Proved)\n", layerIndex, neuronIndex, weightIndex);
        else
            printf("Fail to prove %d %d %d with all masks (Unknown)\n", layerIndex, neuronIndex, weightIndex);
        changeLinear_single_d(nn[tid], trueIndex, neuronIndex, weightIndex, info.weights[layerIndex][neuronIndex][weightIndex]);
    }
    for(int i = 0; i < OMP_NUM_THREADS; i++){
        clean(nn[i]);
    }
    return all_proved_neg;
}

int binaryVerifyDPRres1 = 0;
int binaryVerifyDPRres2 = 1;
//Return 1 if Proved for all sample with [l,r]
int binaryVerify(NeuralNetwork* nn, const nnInfo&info, int layerIndex, int trueIndex, int neuronIndex, int weightIndex, int virtualNeuronNum, const std::vector<double> filipped_sample, int l,int r, int& queries){
    if(l > r)
        return 1;
    ++queries;
    static const char* MESSAGE[]={"Fail to prove","Proved"};
    if(l == r){
        changeLinear_single_d(nn, trueIndex, neuronIndex, weightIndex, filipped_sample[l]);
        int res = test_d(nn, info.input_lower.data(), info.input_upper.data(), info.label, true);
        changeLinear_single_d(nn, trueIndex, neuronIndex, weightIndex, info.weights[layerIndex][neuronIndex][weightIndex]);
        // printf("(Divide and Counter) %s %d %d %d by DeepPoly at [%d]=%lf\n",MESSAGE[res], layerIndex, neuronIndex, weightIndex, l, filipped_sample[l]);
        return res;
    }
    add_variable_weight(nn, trueIndex, neuronIndex, weightIndex, filipped_sample[l], filipped_sample[r], virtualNeuronNum);
    int resAll = test_d(nn, info.input_lower.data(), info.input_upper.data(), info.label, true);
    resetInfo(nn, {trueIndex}, {neuronIndex}, {weightIndex}, {info.weights[layerIndex][neuronIndex][weightIndex]}, virtualNeuronNum);
    // printf("(Divide and Counter) %s %d %d %d by DeepPolyR at [%d, %d]=[%lf, %lf]\n",MESSAGE[resAll], layerIndex, neuronIndex, weightIndex, l, r, filipped_sample[l], filipped_sample[r]);
    double valueMid = (filipped_sample[l] + filipped_sample[r]) / 2;
    int mid = std::upper_bound(filipped_sample.begin(), filipped_sample.end(), valueMid) - filipped_sample.begin() - 1;
    if(l == 0 && r == (int)filipped_sample.size() - 1){
        int res1 = 0;
        add_variable_weight(nn, trueIndex, neuronIndex, weightIndex, filipped_sample[l], filipped_sample[mid], virtualNeuronNum);
        res1 = test_d(nn, info.input_lower.data(), info.input_upper.data(), info.label, true);
        resetInfo(nn, {trueIndex}, {neuronIndex}, {weightIndex}, {info.weights[layerIndex][neuronIndex][weightIndex]}, virtualNeuronNum);
        int res2 = 0;
        add_variable_weight(nn, trueIndex, neuronIndex, weightIndex, filipped_sample[mid + 1], filipped_sample[r], virtualNeuronNum);
        res2 = test_d(nn, info.input_lower.data(), info.input_upper.data(), info.label, true);
        resetInfo(nn, {trueIndex}, {neuronIndex}, {weightIndex}, {info.weights[layerIndex][neuronIndex][weightIndex]}, virtualNeuronNum);
        // printf("(DeepPolyR) %s %d %d %d by DeepPolyR at [%d, %d]=[%lf, %lf]\n",MESSAGE[res1], layerIndex, neuronIndex, weightIndex, l, r, filipped_sample[l], filipped_sample[mid]);
        // printf("(DeepPolyR) %s %d %d %d by DeepPolyR at [%d, %d]=[%lf, %lf]\n",MESSAGE[res2], layerIndex, neuronIndex, weightIndex, mid + 1, r, filipped_sample[mid + 1], filipped_sample[r]);
        binaryVerifyDPRres1 = res1;
        binaryVerifyDPRres2 = res2;
        resAll = 0;
    }
    if(resAll){
        return 1;
    }
    return binaryVerify(nn, info, layerIndex, trueIndex, neuronIndex, weightIndex, virtualNeuronNum, filipped_sample, l, mid, queries) && binaryVerify(nn, info, layerIndex, trueIndex, neuronIndex, weightIndex, virtualNeuronNum, filipped_sample, mid + 1, r, queries);
}

//Return 1 if Proved for all sample with [l,r]
int binaryVerify(NeuralNetwork* nn, const nnInfo&info, int layerIndex, int trueIndex, int neuronIndex, int virtualNeuronNum, const std::vector<double> filipped_sample, int l,int r, int& queries){
    if(l > r)
        return 1;
    ++queries;
    static const char* MESSAGE[]={"Fail to prove","Proved"};
    if(l == r){
        changeBiasSingle(nn, trueIndex, neuronIndex, filipped_sample[l], filipped_sample[r]);
        int res = test_d(nn, info.input_lower.data(), info.input_upper.data(), info.label, true);
        changeBiasSingle(nn, trueIndex, neuronIndex, info.bias[layerIndex][neuronIndex], info.bias[layerIndex][neuronIndex]);
        // printf("(Divide and Counter) %s %d %d (bias) by DeepPoly at [%d]=%lf\n",MESSAGE[res], layerIndex, neuronIndex, l, filipped_sample[l]);
        return res;
    }
    add_variable_bias(nn, trueIndex, neuronIndex, filipped_sample[l], filipped_sample[r]);
    int resAll = test_d(nn, info.input_lower.data(), info.input_upper.data(), info.label, true);
    resetInfo(nn, {trueIndex}, {neuronIndex}, {-1}, {info.bias[layerIndex][neuronIndex]}, virtualNeuronNum);
    // printf("(Divide and Counter) %s %d %d (bias) by DeepPolyR at [%d, %d]=[%lf, %lf]\n",MESSAGE[resAll], layerIndex, neuronIndex, l, r, filipped_sample[l], filipped_sample[r]);
    double valueMid = (filipped_sample[l] + filipped_sample[r]) / 2;
    int mid = std::upper_bound(filipped_sample.begin(), filipped_sample.end(), valueMid) - filipped_sample.begin() - 1;
    if(l == 0 && r == (int)filipped_sample.size() - 1){
        int res1 = 0;
        add_variable_bias(nn, trueIndex, neuronIndex, filipped_sample[l], filipped_sample[mid]);
        res1 = test_d(nn, info.input_lower.data(), info.input_upper.data(), info.label, true);
        resetInfo(nn, {trueIndex}, {neuronIndex}, {-1}, {info.bias[layerIndex][neuronIndex]}, virtualNeuronNum);
        int res2 = 0;
        add_variable_bias(nn, trueIndex, neuronIndex, filipped_sample[mid + 1], filipped_sample[r]);
        res2 = test_d(nn, info.input_lower.data(), info.input_upper.data(), info.label, true);
        resetInfo(nn, {trueIndex}, {neuronIndex}, {-1}, {info.bias[layerIndex][neuronIndex]}, virtualNeuronNum);
        // printf("(DeepPolyR) %s %d %d (bias) by DeepPolyR at [%d, %d]=[%lf, %lf]\n",MESSAGE[res1], layerIndex, neuronIndex, l, r, filipped_sample[l], filipped_sample[mid]);
        // printf("(DeepPolyR) %s %d %d (bias) by DeepPolyR at [%d, %d]=[%lf, %lf]\n",MESSAGE[res2], layerIndex, neuronIndex, mid + 1, r, filipped_sample[mid + 1], filipped_sample[r]);
        binaryVerifyDPRres1 = res1;
        binaryVerifyDPRres2 = res2;
    }
    if(resAll){
        return 1;
    }
    return binaryVerify(nn, info, layerIndex, trueIndex, neuronIndex, virtualNeuronNum, filipped_sample, l, mid, queries) && binaryVerify(nn, info, layerIndex, trueIndex, neuronIndex, virtualNeuronNum, filipped_sample, mid + 1, r, queries);
}
int cnt[2][2][2];

int verifyBinarysearchWeight(const nnInfo&info, int targets_per_layer, int bit_flip_cnt, int&queries) {
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

    std::cout << std::endl;
    printf("!DEBUG: Infer once with no flips");
    std::cout << std::endl;
    int no_flip_proved = test_d(nn[0], info.input_lower.data(), info.input_upper.data(), info.label, true);
    printf("no_flip_proved=%d\n",no_flip_proved);
    printf("!DEBUG: Infer once with no flips Ended\n");  
    if(!no_flip_proved){
        std::cout << "Fail to prove with no flips" << std::endl;
        for(int i = 0; i < OMP_NUM_THREADS; i++)
            clean(nn[i]);
        return 1;
    }

    
    for(int layerIndex = 1; layerIndex < (int)info.weights.size(); ++layerIndex){
        auto thisL = getTargets(layerIndex, info.weights[layerIndex].size(), info.weights[layerIndex][0].size(), targets_per_layer);
        targets.insert(targets.end(), thisL.begin(), thisL.end());
    }
    int all_proved_neg = 0;

    std::cout << "preprocessing end\n"; 
// #pragma omp parallel for num_threads(OMP_NUM_THREADS) 
    for(int i = 0; i < (int)targets.size(); ++i){
        int tid = omp_get_thread_num();
        int layerIndex = targets[i].layerIndex;
        int trueIndex = getTrueAffineIndex(nn[tid], layerIndex, sizes[tid]);
        int neuronIndex = targets[i].neuronIndex;
        int weightIndex = targets[i].weightIndex;
        
        const int fixed_value = std::round(info.weights[layerIndex][neuronIndex][weightIndex] / info.DeltaWs[layerIndex]);
        std::vector<double> flipped_sample;
        flipped_sample.reserve(masks.size());
        for(auto mask : masks){
            double value = flip_bit_mask(fixed_value, mask, info.bit_all) * info.DeltaWs[layerIndex];
            flipped_sample.push_back(value);
        }
        sort(flipped_sample.begin(), flipped_sample.end());
        int res = binaryVerify(nn[tid], info, layerIndex, trueIndex, neuronIndex, weightIndex, virtualNeuronNum, flipped_sample, 0, (int)flipped_sample.size() - 1, queries);   
        int baseline = 1;
        for(auto mask:masks){
            double fipped_val = flip_bit_mask(fixed_value, mask, info.bit_all) * info.DeltaWs[layerIndex];
            changeLinear_single_d(nn[tid], trueIndex, neuronIndex, weightIndex, fipped_val);
            int res = test_d(nn[tid], info.input_lower.data(), info.input_upper.data(), info.label, true);
            changeLinear_single_d(nn[tid], trueIndex, neuronIndex, weightIndex, info.weights[layerIndex][neuronIndex][weightIndex]);
            if(!res){
                baseline = 0;
                break;
            }
        }
        if(baseline){
            // printf("(DeepPoly) Proved %d %d %d with all masks.", layerIndex, neuronIndex, weightIndex);
        }else{
            // printf("(DeepPoly) Fail to prove %d %d %d with all masks.", layerIndex, neuronIndex, weightIndex);
        }
        if(res){
            // printf("(Overall) Proved %d %d %d with all masks. Summary: %d %d %d \n", layerIndex, neuronIndex, weightIndex, res, binaryVerifyDPRres1 && binaryVerifyDPRres2, baseline );
        }else{
            // printf("(Overall) Fail to prove %d %d %d with all masks. Summary: %d %d %d \n", layerIndex, neuronIndex, weightIndex, res, binaryVerifyDPRres1 && binaryVerifyDPRres2, baseline );
            all_proved_neg = 1;
        }
        ++cnt[res][binaryVerifyDPRres1 && binaryVerifyDPRres2][baseline];
        // std::cout << std::endl;
    }
    for(int i = 0; i < OMP_NUM_THREADS; i++){
        clean(nn[i]);
    }
    // std::cout << "flg = " << flg << std::endl;
    return all_proved_neg;
}


int verifyBinarysearchBias(const nnInfo&info, int targets_per_layer, int bit_flip_cnt, int&queries) {
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

    std::cout << std::endl;
    printf("!DEBUG: Infer once with no flips");
    std::cout << std::endl;
    int no_flip_proved = test_d(nn[0], info.input_lower.data(), info.input_upper.data(), info.label, true);
    printf("no_flip_proved=%d\n",no_flip_proved);
    printf("!DEBUG: Infer once with no flips Ended\n");  
    if(!no_flip_proved){
        std::cout << "Fail to prove with no flips" << std::endl;
        for(int i = 0; i < OMP_NUM_THREADS; i++)
            clean(nn[i]);
        return 1;
    }

    
    for(int layerIndex = 1; layerIndex < (int)info.weights.size(); ++layerIndex){
        int this_layer_targets = std::min(targets_per_layer, (int)info.weights[layerIndex].size());
        if(this_layer_targets < targets_per_layer)
            std::cout << "Warning: layer " << layerIndex << " has less than " << targets_per_layer << " targets" << std::endl;
        auto thisL = getTargets(layerIndex, info.weights[layerIndex].size(), 1, this_layer_targets);
        targets.insert(targets.end(), thisL.begin(), thisL.end());
    }
    int all_proved_neg = 0;

    std::cout << "preprocessing end\n"; 
// #pragma omp parallel for num_threads(OMP_NUM_THREADS) 
    for(int i = 0; i < (int)targets.size(); ++i){
        int tid = omp_get_thread_num();
        int layerIndex = targets[i].layerIndex;
        int trueIndex = getTrueAffineIndex(nn[tid], layerIndex, sizes[tid]) + 1;
        int neuronIndex = targets[i].neuronIndex;
        
        const int fixed_value = std::round(info.bias[layerIndex][neuronIndex] / info.DeltaWs[layerIndex]);
        std::vector<double> flipped_sample;
        flipped_sample.reserve(masks.size());
        for(auto mask : masks){
            double value = flip_bit_mask(fixed_value, mask, info.bit_all) * info.DeltaWs[layerIndex];
            flipped_sample.push_back(value);
        }
        sort(flipped_sample.begin(), flipped_sample.end());
        int res = binaryVerify(nn[tid], info, layerIndex, trueIndex, neuronIndex, virtualNeuronNum, flipped_sample, 0, (int)flipped_sample.size() - 1, queries);
        int baseline = 1;
        for(auto mask: masks){
            double fipped_val = flip_bit_mask(fixed_value, mask, info.bit_all) * info.DeltaWs[layerIndex];
            changeBiasSingle(nn[tid], trueIndex, neuronIndex, fipped_val, fipped_val);
            int res = test_d(nn[tid], info.input_lower.data(), info.input_upper.data(), info.label, true);
            changeBiasSingle(nn[tid], trueIndex, neuronIndex, info.bias[layerIndex][neuronIndex], info.bias[layerIndex][neuronIndex]);
            if(!res){
                baseline = 0;
                break;
            }
        }
        if(res){
            // printf("(Overall) Proved %d %d (bias) with all masks. Summary: %d %d %d \n", layerIndex, neuronIndex, res, binaryVerifyDPRres1, binaryVerifyDPRres2);
        }else{
            // printf("(Overall) Fail to prove %d %d (bias) with all masks. Summary: %d %d %d \n", layerIndex, neuronIndex, res, binaryVerifyDPRres1, binaryVerifyDPRres2);
            all_proved_neg = 1;
        }
        ++cnt[res][binaryVerifyDPRres1 && binaryVerifyDPRres2][baseline];
        // std::cout << std::endl;
    }
    for(int i = 0; i < OMP_NUM_THREADS; i++){
        clean(nn[i]);
    }
    // std::cout << "flg = " << flg << std::endl;
    return all_proved_neg;
}


int main(int argc, char *argv[]){    
    srand(123);
    //set precision of cout
    
    
    if(argc < 5){
        std::cout << "Usage: " << argv[0] << " <json file> <method (baseline,binarysearch)> <targets_per_layer> <bit_flip_cnt>" << std::endl;
        return 1;
    }

    int res = validate();
    if(res != 0){
        std::cout << "Validation failed. Please check your environment!" << std::endl;
        return 2;
    }
    // [2 8] 
    // [2 8] 
    // [-4 3.28626e-14] 
    // [-1.19904e-14 4] 
    // [2 8] 
    // [-8 -1] 
    // Should output this if you uncomment the res.print in network.cu:run
    // And uncomment evaluateAffine<T>(res, AlwaysKeep<T>(), layers.size() - 1, true, sound, finalA);

    nnInfo info;
    if(readJson(argv[1], info) != 0){
        std::cout << "Error reading json" << std::endl;
        return 1;
    }
    assert(std::string(argv[2]) == "baseline" || std::string(argv[2]) == "binarysearch" || std::string(argv[2]) == "binarysearch_bias" || std::string(argv[2]) == "binarysearch_all");
    int targets_per_layer = std::stoi(argv[3]);
    int bit_flip_cnt = std::stoi(argv[4]);
    int queries = 0;
    printf("json file: %s\n",argv[1]);
    printf("method: %s\n",argv[2]);
    printf("targets_per_layer: %d\n",targets_per_layer);
    printf("bit_flip_cnt: %d\n",bit_flip_cnt);
    auto start_time = std::chrono::system_clock::now();
    // int res2 = std::string(argv[2]) == "baseline" ? verifyBaselineWeight(info, targets_per_layer, bit_flip_cnt, queries) : verifyBinarysearchWeight(info, targets_per_layer, bit_flip_cnt, queries);
    int res2 = -1;
    if(std::string(argv[2]) == "baseline")
        res2 = verifyBaselineWeight(info, targets_per_layer, bit_flip_cnt, queries);
    else if(std::string(argv[2]) == "binarysearch")
        res2 = verifyBinarysearchWeight(info, targets_per_layer, bit_flip_cnt, queries);
    else if(std::string(argv[2]) == "binarysearch_bias")
        res2 = verifyBinarysearchBias(info, targets_per_layer, bit_flip_cnt, queries);
    else if(std::string(argv[2]) == "binarysearch_all"){
        int res1 = verifyBinarysearchWeight(info, targets_per_layer, bit_flip_cnt, queries);
        int temp = verifyBinarysearchBias(info, targets_per_layer, bit_flip_cnt, queries);
        res2 = res1 || temp;
    }
    else{
        std::cout << "Unknown method" << std::endl;
        return 1;
    }
    auto end_time = std::chrono::system_clock::now();
    std::cout << "Elapsed Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1e3 << std::endl;    
    std::cout << "flg =" << res2 << std::endl;
    std::cout << "all_proved =" << !res2 << std::endl;
    std::cout << "queries = " << queries << std::endl;
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            for(int k = 0; k < 2; ++k)
                std::cout << "cnt[" << i << "][" << j << "][" << k << "] = " << cnt[i][j][k] << std::endl;
    return 0;
}