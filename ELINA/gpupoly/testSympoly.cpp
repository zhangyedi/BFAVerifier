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
        nn[i] = createNN(ActivationType::ReLU, layerSizes, weights,bias,virtualNeuronNum,trueLayerSize);
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
                    int l,int r, int& queries, int degrade, int norecursive,
                    int addaptive_union, int at_first_layer_and_union_sinlge, int&LargestProvedIndex){
    // adaptive union in recursive mode:
    // union the current bound to the recorded bound if 
    // 1. safe 2. left return not proved and the current is the highest node that is without a right child
    if(l > r)
        return 1;
    ++queries;
    static const char* MESSAGE[]={"Fail to prove","Proved"};
    const double original_value = weightIndex == -1 ? info.bias[layerIndex][neuronIndex] 
                                : info.weights[layerIndex][neuronIndex][weightIndex];
    if(l == r){
        char mode_before = 0;
        if(at_first_layer_and_union_sinlge){
            mode_before = getRecoredConcreteBoundsMode(nn);
            enableRecordConcreteBounds(nn, 'u');
        }
        add_variable_para(nn, trueIndex, neuronIndex, weightIndex, filipped_sample[l], filipped_sample[r], virtualNeuronNum);
        int res = test_d(nn, info.input_lower.data(), info.input_upper.data(), info.label, true);
        resetInfo(nn, {trueIndex}, {neuronIndex}, {weightIndex}, {original_value}, virtualNeuronNum);
        if(at_first_layer_and_union_sinlge)
            enableRecordConcreteBounds(nn, mode_before);
        // printf("(Divide and Counter) %s %d %d %d by SymPoly at [%d,%d]=[%lf,%lf]\n",MESSAGE[res], layerIndex, neuronIndex, weightIndex, l, r, filipped_sample[l], filipped_sample[r]);
        if(res)
            LargestProvedIndex = std::max(LargestProvedIndex, r);
        return res;
    }
    int resAll = 0;
    if(degrade != 1){
        add_variable_para(nn, trueIndex, neuronIndex, weightIndex, filipped_sample[l], filipped_sample[r], virtualNeuronNum);
        resAll = test_d(nn, info.input_lower.data(), info.input_upper.data(), info.label, true);
        // printf("(Divide and Counter) %s %d %d %d by SymPoly at [%d,%d]=[%lf,%lf]\n",MESSAGE[resAll], layerIndex, neuronIndex, weightIndex, l, r, filipped_sample[l], filipped_sample[r]);
        resetInfo(nn, {trueIndex}, {neuronIndex}, {weightIndex}, {original_value}, virtualNeuronNum);
    }
    double valueMid = (filipped_sample[l] + filipped_sample[r]) / 2;
    int mid = std::upper_bound(filipped_sample.begin(), filipped_sample.end(), valueMid) - filipped_sample.begin() - 1;
    if(resAll){
        LargestProvedIndex = std::max(LargestProvedIndex, r);
        return 1;
    }
    if(norecursive)
        return 0;
    int resL = binaryVerify(nn, info, layerIndex, trueIndex, neuronIndex, weightIndex, virtualNeuronNum, filipped_sample, l, mid, queries, degrade, norecursive, 0, 0, LargestProvedIndex);

    if (degrade != 1 && addaptive_union && resL == 0){
        char mode_before = getRecoredConcreteBoundsMode(nn);
        enableRecordConcreteBounds(nn, 'u');
        add_variable_para(nn, trueIndex, neuronIndex, weightIndex, filipped_sample[l], filipped_sample[r], virtualNeuronNum);
        resAll = test_d(nn, info.input_lower.data(), info.input_upper.data(), info.label, true);
        resetInfo(nn, {trueIndex}, {neuronIndex}, {weightIndex}, {original_value}, virtualNeuronNum);
        enableRecordConcreteBounds(nn, mode_before);
        // printf("(Divide and Counter) %s %d %d %d by SymPoly at [%d,%d]=[%lf,%lf]\n",MESSAGE[resAll], layerIndex, neuronIndex, weightIndex, l, r, filipped_sample[l], filipped_sample[r]);

        return 0;
    }

    return resL && binaryVerify(nn, info, layerIndex, trueIndex, neuronIndex, weightIndex, virtualNeuronNum, filipped_sample, mid + 1, r, queries, degrade, norecursive, addaptive_union, 0, LargestProvedIndex);
}

#define OMP_NUM_THREADS 1
int verifyBFA_RA(const nnInfo &info, int targets_per_layer, int bit_flip_cnt, int &queries, Method method, int record_union_bounds){
    int all_proved_neg = 0;
    NeuralNetwork *nn[OMP_NUM_THREADS];
    size_t sizes[OMP_NUM_THREADS];
    int virtualNeuronNum = bit_flip_cnt;
    for(int i = 0; i < OMP_NUM_THREADS; i++){
        nn[i] = createNN(info.actType, info.layerSizes, info.weights, info.bias, virtualNeuronNum, sizes[i]);
        if(!nn[i]){
            std::cout << "Error creating network" << std::endl;
            return -1;
        }
    }
    std::vector<target> targets;
    std::vector<int> masks = getBitMask(info.bit_all, bit_flip_cnt);


    if (method == Method::INTEGRATE_TEST){
        std::cout << "INTEGRATE_TEST" << std::endl;
        for(int i = 0; i < OMP_NUM_THREADS; ++i)
            enableIntegrateTestSetter(nn[i], 1);
        for(int i = 0; i < OMP_NUM_THREADS; ++i)
            enableRecordConcreteBounds(nn[i], 'o');
    }
    
    if (record_union_bounds){
        for(int i = 0; i < OMP_NUM_THREADS; i++){
            enableRecordConcreteBounds(nn[i], 'o');
            //set to override mode at the first deeppoly
        }
    }
    printf("!DEBUG: Infer once with no flips\n");
    int no_flip_proved = test_d(nn[0], info.input_lower.data(), info.input_upper.data(), info.label, true);
    if(!no_flip_proved){
        std::cout << "Fail to prove with no flips" << std::endl;
        for(int i = 0; i < OMP_NUM_THREADS; i++){
            clean(nn[i]);
        }
        return -1;
    }
    else printf("!DEBUG: Infer once with no flips. Proved\n");
    
    if(record_union_bounds && method == Method::INTEGRATE_TEST){
        if (method == Method::INTEGRATE_TEST)
            printf("[integrate test] SymPoly (GPU) before all flipping cases\n");
        for(int i = 1; i < (int)info.weights.size(); i++){
            if (method == Method::INTEGRATE_TEST)
                printf("[Integrate test] ");
            printf("[Union of All Fault] ");
            printRecorededBounds(nn[0], i, sizes[0], ("on affine layer " + std::to_string(i)).c_str());
            // if (i != (int) info.weights.size() - 1){
            //     if (method == Method::INTEGRATE_TEST)
            //         printf("[Integrate test] ");
            //     printf("[Intermediate Bounds] ");
            //     printPostactivationBounds(nn[0], i, sizes[0], ("on post value " + std::to_string(i)).c_str());
            // }
        }
    }

    if (record_union_bounds){
        if (method == Method::INTEGRATE_TEST){
            for(int i = 0; i < OMP_NUM_THREADS; i++){
                enableRecordConcreteBounds(nn[i], 'u');
                //set to union if safe mode after first deeppoly
            }
        }else{
            for(int i = 0; i < OMP_NUM_THREADS; i++){
                enableRecordConcreteBounds(nn[i], 'i');
                //set to union mode after first deeppoly
            }
        }
    }

    static std::vector<int> neurons_to_test = {0,1,2,3,4}; 
    static std::vector<int> weight_to_test = {0,1,2,3,4,-1}; // contains -1 for bias 

    for(int layerIndex = 1; layerIndex < (int)info.weights.size(); ++layerIndex){
        if (method == Method::INTEGRATE_TEST){
            for(int neuron_id: neurons_to_test)
                for(int weight_id: weight_to_test)
                    targets.emplace_back(target{layerIndex, neuron_id, weight_id});
            continue;
        }
        //add weights
        auto thisL = getTargets(layerIndex, info.weights[layerIndex].size(), info.weights[layerIndex][0].size(), targets_per_layer);
        //add bias
        auto thisB = getTargets(layerIndex, info.weights[layerIndex].size(), 1, targets_per_layer);
        for(auto& tar:thisB) tar.weightIndex = -1;
        targets.insert(targets.end(), thisL.begin(), thisL.end());
        targets.insert(targets.end(), thisB.begin(), thisB.end());

    }
    std::cout << "targets size: " << targets.size() << std::endl;


    printf("!DEBUG: LOG OPTION: OMIT PROVED. PRINT FAILED.\n");
    const int OMIT_PROVED = 1;
    int DEBUG_REACH_1 = 5000;
    for(int i = 0; i < (int)targets.size(); ++i){
        int tid = omp_get_thread_num();
        int layerIndex = targets[i].layerIndex;
        int weightIndex = targets[i].weightIndex;
        int trueIndex = 
            weightIndex == -1?
            getTrueAffineIndex(nn[tid], layerIndex, sizes[tid]) + 1:
            getTrueAffineIndex(nn[tid], layerIndex, sizes[tid]);
        int neuronIndex = targets[i].neuronIndex;
        //! debug 2 0 0 
        // if (!(layerIndex == 2 && neuronIndex == 0 && weightIndex == 0)){
        //     continue;
        // }
        auto print_intermediate_bounds = [&](){
                if(record_union_bounds){
                    for(int i = 1; i < (int)info.weights.size(); i++){
                        if (method == Method::INTEGRATE_TEST)
                            printf("[Integrate test] ");
                        printf("[Intermediate Bounds] ");
                        printIntermediateBounds(nn[0], i, sizes[0], ("on affine layer " + std::to_string(i)).c_str());
                        if (i != (int) info.weights.size() - 1){
                            if (method == Method::INTEGRATE_TEST)
                                printf("[Integrate test] ");
                            printf("[Intermediate Bounds] ");
                            printPostactivationBounds(nn[0], i, sizes[0], ("on post value " + std::to_string(i)).c_str());
                        }
                    }
                }
        };

        if(i % 5000 == 0){
            //flush stdout
            std::cout << "Currently processed " << i << " targets, at layer " 
            << layerIndex << " with neuron " << neuronIndex << " and weight " 
            << weightIndex << std::endl;
        }
        const double paraValue = weightIndex == -1 ? 
                info.bias[layerIndex][neuronIndex] 
                : info.weights[layerIndex][neuronIndex][weightIndex];
        const int fixed_value = std::round(
                            paraValue / info.DeltaWs[layerIndex] 
                            );

        if(method == Method::INTEGRATE_TEST){
            printf("[integrate test] SymPoly (GPU) flipping on %d %d %s, Flt=%.4f Int=%d\n", 
                    layerIndex, neuronIndex, 
                    weightIndex == -1 ? "bias" : std::to_string(weightIndex).c_str(),
                    paraValue, fixed_value);
        }

        std::vector<double> flipped_samples;
        flipped_samples.reserve(masks.size());
        for(auto mask:masks){
            double value = flip_bit_mask(fixed_value, mask, info.bit_all)
                            * info.DeltaWs[layerIndex];
            flipped_samples.push_back(value);
        }
        sort(flipped_samples.begin(), flipped_samples.end());
        int res = 0;
        if(res == 0){
            auto ra_binary = [&](){
                int FurthestProvedIndex = -1;
                // if (layerIndex == 1 && neuronIndex == 5 && weightIndex == 2){
                //     enableIntegrateTestSetter(nn[tid], 1);
                //     enableRecordConcreteBounds(nn[tid], 'o');
                //     std::cout << "flt " << flipped_samples[0] << ' ' << flipped_samples.back() << std::endl;
                //     printf("[debug] SymPoly (GPU) flipping on %d %d %s, Flt=%.4f Int=%d\n", 
                //     layerIndex, neuronIndex, 
                //     weightIndex == -1 ? "bias" : std::to_string(weightIndex).c_str(),
                //     paraValue, fixed_value);
                // }
                // int res3 = binaryVerify(nn[tid], info, layerIndex, trueIndex,
                //                 neuronIndex, weightIndex, virtualNeuronNum, flipped_samples, 
                //                 0, (int)flipped_samples.size() - 1, queries, method == BASELINE, 1, record_union_bounds, 0, FurthestProvedIndex);
                // if(res3)
                //     return true;
                // if (layerIndex == 1 && neuronIndex == 5 && weightIndex == 2){
                //     enableIntegrateTestSetter(nn[tid], 0);
                //     enableRecordConcreteBounds(nn[tid], 'i');
                //     for(int _i = 1; _i < (int)info.weights.size(); _i++){
                //         if (method == Method::INTEGRATE_TEST)
                //             printf("[On this debug] ");
                //         printf("[On this debug] ");
                //         printRecorededBounds(nn[0], _i, sizes[0], ("on affine layer " + std::to_string(_i)).c_str());
                //     }
                // }
                int last_less_eq_0 = std::lower_bound(flipped_samples.begin(), flipped_samples.end(), 0) - flipped_samples.begin();
                last_less_eq_0 -= 1;
                int res1 = binaryVerify(nn[tid], info, layerIndex, trueIndex,
                                neuronIndex, weightIndex, virtualNeuronNum, flipped_samples, 
                                0, last_less_eq_0, queries, method == BASELINE, 0, record_union_bounds, 0, FurthestProvedIndex);
                if(!res1){
                    if (record_union_bounds){
                        char past_mode = getRecoredConcreteBoundsMode(nn[tid]);
                        enableRecordConcreteBounds(nn[tid], 'u');
                        binaryVerify(nn[tid], info, layerIndex, trueIndex,
                                    neuronIndex, weightIndex, virtualNeuronNum, flipped_samples, 
                                    FurthestProvedIndex + 1, last_less_eq_0, queries, method == BASELINE, 1, 0, 0, FurthestProvedIndex);
                        binaryVerify(nn[tid], info, layerIndex, trueIndex,
                                    neuronIndex, weightIndex, virtualNeuronNum, flipped_samples, 
                                    last_less_eq_0 + 1, (int)flipped_samples.size() - 1, queries, method == BASELINE, 1, 0, 0, FurthestProvedIndex);
                        enableRecordConcreteBounds(nn[tid], past_mode);
                    }
                    return false;
                }
                int res2 = binaryVerify(nn[tid], info, layerIndex, trueIndex,
                                neuronIndex, weightIndex, virtualNeuronNum, flipped_samples, 
                                last_less_eq_0 + 1, (int)flipped_samples.size() - 1, queries, method == BASELINE, 0, record_union_bounds, record_union_bounds, FurthestProvedIndex);
                if (record_union_bounds){
                        // std::cout << "FurthestIndex = " << FurthestProvedIndex << std::endl;
                        char past_mode = getRecoredConcreteBoundsMode(nn[tid]);
                        enableRecordConcreteBounds(nn[tid], 'u');
                        binaryVerify(nn[tid], info, layerIndex, trueIndex,
                                    neuronIndex, weightIndex, virtualNeuronNum, flipped_samples, 
                                    FurthestProvedIndex + 1, last_less_eq_0, queries, method == BASELINE, 1, 0, 0, FurthestProvedIndex);
                        binaryVerify(nn[tid], info, layerIndex, trueIndex,
                                    neuronIndex, weightIndex, virtualNeuronNum, flipped_samples, 
                                    last_less_eq_0 + 1, (int)flipped_samples.size() - 1, queries, method == BASELINE, 1, 0, 0, FurthestProvedIndex);
                        enableRecordConcreteBounds(nn[tid], past_mode);
                }
                return res1 && res2;
            };

            auto ra_wo_binary_integreate_test = [&](){
                int FurthestProvedIndex = -1;
                // int res3 = binaryVerify(nn[tid], info, layerIndex, trueIndex,
                //                 neuronIndex, weightIndex, virtualNeuronNum, flipped_samples, 
                //                 0, (int)flipped_samples.size() - 1, queries, 
                //                 0, 1, 0, 0, FurthestProvedIndex);
                // if(res3 && method != Method::INTEGRATE_TEST)
                //     return true;
                // print_intermediate_bounds();
                int last_less_eq_0 = std::lower_bound(flipped_samples.begin(), flipped_samples.end(), 0) - flipped_samples.begin();
                last_less_eq_0 -= 1;
                if (method == Method::INTEGRATE_TEST)
                    printf("[Integrate test] SymWeight=[%.4f %.4f]\t ", flipped_samples[0], flipped_samples[last_less_eq_0]);
                int res1 = binaryVerify(nn[tid], info, layerIndex, trueIndex,
                                neuronIndex, weightIndex, virtualNeuronNum, flipped_samples, 
                                0, last_less_eq_0, queries, 0, 1, 0, 0, FurthestProvedIndex);
                print_intermediate_bounds();
                if (method != Method::INTEGRATE_TEST && !res1)
                    return false;
                if (method == Method::INTEGRATE_TEST)
                    printf("[Integrate test] SymWeight=[%.4f %.4f]\t ", flipped_samples[last_less_eq_0+1], flipped_samples[(int)flipped_samples.size() - 1]);
                int res2 = binaryVerify(nn[tid], info, layerIndex, trueIndex,
                                neuronIndex, weightIndex, virtualNeuronNum, flipped_samples, 
                                last_less_eq_0 + 1, (int)flipped_samples.size() - 1, queries, 0, 1, 0, 0, FurthestProvedIndex);  
                print_intermediate_bounds();
                return res1 && res2;
            };

            auto ra_wo_binary = [&](){
                int FurthestProvedIndex = -1;
                // int res3 = binaryVerify(nn[tid], info, layerIndex, trueIndex,
                //                 neuronIndex, weightIndex, virtualNeuronNum, flipped_samples, 
                //                 0, (int)flipped_samples.size() - 1, queries, 
                //                 0, 1, 0, 0, FurthestProvedIndex);
                // // std::cout << "res3=" <<  res3 << std::endl;
                // if(res3)
                //     return true;
                int last_less_eq_0 = std::lower_bound(flipped_samples.begin(), flipped_samples.end(), 0) - flipped_samples.begin();
                last_less_eq_0 -= 1;

                int res1 = binaryVerify(nn[tid], info, layerIndex, trueIndex,
                                neuronIndex, weightIndex, virtualNeuronNum, flipped_samples, 
                                0, last_less_eq_0, queries, 0, 1, 0, 0, FurthestProvedIndex);
                // std::cout << "res1=" << res1 << ' ' << last_less_eq_0  << ' ' << (int)flipped_samples.size() - 1 << std::endl; 
                if(!res1){
                    if(record_union_bounds){
                        enableRecordConcreteBounds(nn[tid], 'u');
                        binaryVerify(nn[tid], info, layerIndex, trueIndex,
                                    neuronIndex, weightIndex, virtualNeuronNum, flipped_samples, 
                                    0, (int)flipped_samples.size() - 1, queries, 0, 1, 0, 0, FurthestProvedIndex);
                        enableRecordConcreteBounds(nn[tid], 'i');
                    }
                    return false;
                }

                if (record_union_bounds)
                    enableRecordConcreteBounds(nn[tid], 'u');
                int res2 = binaryVerify(nn[tid], info, layerIndex, trueIndex,
                                neuronIndex, weightIndex, virtualNeuronNum, flipped_samples, 
                                last_less_eq_0 + 1, (int)flipped_samples.size() - 1, queries, 0, 1, 0, record_union_bounds, FurthestProvedIndex);  
                if (record_union_bounds)
                    enableRecordConcreteBounds(nn[tid], 'i');
                // std::cout << "res2=" << res2 << ' ' << last_less_eq_0 + 1 << ' ' << (int)flipped_samples.size() - 1 <<  std::endl; 

                return res1 && res2;
            };

            auto ra_one_sym = [&](){
                int FurthestProvedIndex = -1;
                if (method == Method::INTEGRATE_TEST)
                    printf("[Integrate test] SymWeight=[%.4f %.4f]\t ", flipped_samples[0], flipped_samples[(int)flipped_samples.size() - 1]);
                char past = getRecoredConcreteBoundsMode(nn[tid]);
                if (record_union_bounds)
                    enableRecordConcreteBounds(nn[tid], 'u');
                int res1 = binaryVerify(nn[tid], info, layerIndex, trueIndex,
                    neuronIndex, weightIndex, virtualNeuronNum, flipped_samples, 
                    0, (int)flipped_samples.size() - 1, queries, 0, 1, 0, 0, FurthestProvedIndex);
                if (record_union_bounds){
                    enableRecordConcreteBounds(nn[tid], past);
                    if (method == Method::INTEGRATE_TEST)
                        print_intermediate_bounds();
                }
                return res1;
            };

            if (method == Method::BINARYSEARCH || method == Method::BASELINE){
                res = ra_binary();
            }
            else if(method == Method::BFA_RA_WO_BINARY){
                res = ra_wo_binary();
            }else if(method == Method::BFA_ONE_INTERVAL){
                res = ra_one_sym();
            }else if(method == Method::INTEGRATE_TEST){
                res = ra_wo_binary_integreate_test();
                // res = ra_one_sym() || res;
            }
            else {// impossible
                assert(0);
            }
        }
        if(res){
            if (OMIT_PROVED)
                continue;
            if(weightIndex != -1)
                printf("(Overall) Proved %d %d %d with all masks. Summary: %d %d %d \n", layerIndex, neuronIndex, weightIndex, res, -1, -1 );
            else
                printf("(Overall) Proved %d %d (bias) with all masks. Summary: %d %d %d \n", layerIndex, neuronIndex, res, -1, -1);

        }else{
            if(weightIndex != -1)
                printf("(Overall) Fail to prove %d %d %d with all masks. Summary: %d %d %d \n", layerIndex, neuronIndex, weightIndex, res, -1, -1);
            else
                printf("(Overall) Fail to prove %d %d (bias) with all masks. Summary: %d %d %d \n", layerIndex, neuronIndex, res, -1, -1);
            all_proved_neg = 1;
            if(targets_per_layer == -2)
                break;
        }
    }
    if(record_union_bounds){
        if (method == Method::INTEGRATE_TEST)
            printf("[integrate test] SymPoly (GPU) save the union of all flipping cases\n");
        for(int i = 1; i < (int)info.weights.size(); i++){
            if (method == Method::INTEGRATE_TEST)
                printf("[Integrate test] ");
            printf("[Union of All Fault] ");
            printRecorededBounds(nn[0], i, sizes[0], ("on affine layer " + std::to_string(i)).c_str());
        }
    }
    for(int i = 0; i < OMP_NUM_THREADS; i++){
        clean(nn[i]);
    }
    return all_proved_neg;
}

#define OMP_NUM_THREADS 1

int verifyReLUFault(nnInfo info, int targets_per_layer, int bit_flip_cnt, int &queries){
    int all_proved_neg = 0;
    NeuralNetwork *nn[OMP_NUM_THREADS];
    size_t sizes[OMP_NUM_THREADS];
    int virtualNeuronNum = bit_flip_cnt;
    for(int i = 0; i < OMP_NUM_THREADS; i++){
        nn[i] = createNN(info.actType, info.layerSizes, info.weights, info.bias, virtualNeuronNum, sizes[i]);
        if(!nn[i]){
            std::cout << "Error creating network" << std::endl;
            return -1;
        }
    }
    std::vector<target> targets;
    std::vector<int> masks = getBitMask(info.bit_all, bit_flip_cnt);
    printf("!DEBUG: RELU SKIP VERIFICATION\n");
    printf("!DEBUG: Infer once with no flips\n");
    info.input_lower.resize(info.input_lower.size() + 2, 0);
    info.input_upper.resize(info.input_upper.size() + 2, 0);
    int no_flip_proved = test_d(nn[0], info.input_lower.data(), info.input_upper.data(), info.label, true);
    if(!no_flip_proved){
        std::cout << "Fail to prove with no flips" << std::endl;
        for(int i = 0; i < OMP_NUM_THREADS; i++){
            clean(nn[i]);
        }
        return 1;
    }
    printf("!DEBUG: Infer once with no flips. Proved\n");
    printf("!DEBUG: LOG OPTION: OMIT PROVED. PRINT FAILED.\n");
    const int OMIT_PROVED = 1;
    for(int layerIndex = 2; layerIndex < (int)info.weights.size(); ++layerIndex){
        //add bias
        auto thisR = getTargets(layerIndex, info.weights[layerIndex].size(), 1, targets_per_layer);
        for(auto& tar:thisR) tar.weightIndex = -2; // indicate ReLU skip
        targets.insert(targets.end(), thisR.begin(), thisR.end());
    }
    for(int i = 0; i < (int)targets.size(); ++i){
        int tid = omp_get_thread_num();
        int layerIndex = targets[i].layerIndex;
        int weightIndex = targets[i].weightIndex;
        int trueIndex = 
            weightIndex == -1?
            getTrueAffineIndex(nn[tid], layerIndex, sizes[tid]) + 1:
            getTrueAffineIndex(nn[tid], layerIndex, sizes[tid]); // weightIndex >=0 or weightIndex = -1 share position for relu injects
        int neuronIndex = targets[i].neuronIndex;
        
        if(add_relu_skip(nn[tid], trueIndex, neuronIndex, virtualNeuronNum) != 0){
            std::cout << "Error adding ReLU Skip info" << std::endl;
            for(int i = 0; i < OMP_NUM_THREADS; i++){
                clean(nn[i]);
            }
            return -1;
        }
        int res = test_d(nn[tid], info.input_lower.data(), info.input_upper.data(), info.label, true);
        ++queries;
        
        if(resetInfo(nn[tid], {trueIndex}, {neuronIndex}, {-2}, {0}, virtualNeuronNum) != 0){
            std::cout << "Error resetting ReLU info" << std::endl;
            for(int i = 0; i < OMP_NUM_THREADS; i++){
                clean(nn[i]);
            }
            return -1;
        }

        if(res){
            if (OMIT_PROVED)
                continue;
            if (weightIndex == -2)
                printf("(Overall) Proved %d %d (ReLU-Skip) with all masks. Summary: %d %d %d \n", layerIndex, neuronIndex, res, -1, -1);
        }else{
            if (weightIndex == -2)
                printf("(Overall) Fail to prove %d %d (ReLU-Skip) with all masks. Summary: %d %d %d \n", layerIndex, neuronIndex, res, -1, -1);
            all_proved_neg = 1;
        }
        if(i % 100 == 0){
            //flush stdout
            std::cout << "Currently processed " << i << " targets, at layer " 
            << layerIndex << " with neuron " << neuronIndex << " and weight " 
            << weightIndex << std::endl;
        }
    }
    for(int i = 0; i < OMP_NUM_THREADS; i++){
        clean(nn[i]);
    }
    return all_proved_neg;
}

int main(int argc, char *argv[]){    
    srand(123);
    //set precision of cout
    // validate();
    // printf("Finished Validating\n-------------\n");

    if(argc < 5){
        std::cout << "Usage: " << argv[0] << " <json file> <method (binarysearch_all, baseline, relu_skip, bfa_ra_wo_binary, bfa_ra_one_interval, integrate_test)> <targets_per_layer> <bit_flip_cnt> <record_union_bounds (optional)>" << std::endl;
        std::cout << "Usage: " << argv[0] << " <weight file> <input json> <method (binarysearch_all, baseline, relu_skip, bfa_ra_wo_binary, bfa_ra_one_interval, integrate_test)> <targets_per_layer> <bit_flip_cnt> <eps> <record_union_bounds (optional)>" << std::endl;
        return 1;
    }
    std::string jsonFile = argv[1];
    std::string method = "";
    std::string inputJson = "";
    int targets_per_layer = 0;
    int bit_flip_cnt = 0;
    int queries = 0;
    double radius = 0;
    int record_union_bounds = 0;
    if(argc == 5 || argc == 6){
        method = argv[2];
        targets_per_layer = std::stoi(argv[3]);
        bit_flip_cnt = std::stoi(argv[4]);
        if (argc == 6)
            record_union_bounds = std::string(argv[5]) == "record_union_bounds";
    }else if (argc == 7 || argc == 8){
        inputJson = argv[2];
        method = argv[3];
        targets_per_layer = std::stoi(argv[4]);
        bit_flip_cnt = std::stoi(argv[5]);
        radius = std::stod(argv[6]);
        if(argc == 8)
            record_union_bounds = std::string(argv[7]) == "record_union_bounds";
    }else{
        std::cout << "Invalid number of arguments" << std::endl;
        return 1;
    }

    // int res = validate();
    // if(res != 0){
    //     std::cout << "Validation failed. Please check your environment!" << std::endl;
    //     return 2;
    // }

    nnInfo info;
    if(readJson(jsonFile, info) != 0){
        std::cout << "Error reading json " + jsonFile << std::endl;
        return 1;
    }

    if(inputJson != ""){
        if(setInputFromPara(inputJson, radius, info) != 0){
            std::cout << "Error reading json for input! " + inputJson << std::endl;
            return 1;
        }
        std::cout << "input json path" + inputJson << std::endl;
    }

    assert(method == "binarysearch_all" || method == "relu_skip" || method == "baseline" 
            || method == "bfa_ra_wo_binary" || method == "bfa_ra_one_interval"
            || method == "integrate_test"
        );

    printf("json file: %s\n",jsonFile.c_str());
    printf("method: %s\n",method.c_str());
    printf("targets_per_layer: %d\n",targets_per_layer);
    printf("bit_flip_cnt: %d\n",bit_flip_cnt);
    printf("input file (optional if contained in json file): %s\n",inputJson.c_str());
    printf("radius (optional if contained in json file): %lf\n",radius);
    printf("record_union_bounds: %d\n",record_union_bounds);
    printf("actType (main): %d\n",info.actType);
    printf("label (main): %d\n",info.label);

    auto start_time = std::chrono::system_clock::now();
    int res2 = 0;
    if(method == "binarysearch_all")
        res2 = verifyBFA_RA(info, targets_per_layer, bit_flip_cnt, queries, BINARYSEARCH, record_union_bounds);
    else if(method == "baseline")
        res2 = verifyBFA_RA(info, targets_per_layer, bit_flip_cnt, queries, BASELINE, record_union_bounds);
    else if(method == "relu_skip")
        res2 = verifyReLUFault(info, targets_per_layer, bit_flip_cnt, queries);
    else if(method == "bfa_ra_wo_binary")
        res2 = verifyBFA_RA(info, targets_per_layer, bit_flip_cnt, queries, BFA_RA_WO_BINARY, record_union_bounds);
    else if(method == "bfa_ra_one_interval")
        res2 = verifyBFA_RA(info, targets_per_layer, bit_flip_cnt, queries, BFA_ONE_INTERVAL, record_union_bounds);
    else if(method == "integrate_test")
        res2 = verifyBFA_RA(info, targets_per_layer, bit_flip_cnt, queries, INTEGRATE_TEST, record_union_bounds);
    else{
        std::cout << "Invalid method" << std::endl;
        return 1;
    }

    auto end_time = std::chrono::system_clock::now();
    std::cout << "Elapsed Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1e3 << std::endl;
    std::cout << "flg =" << res2 << std::endl;
    std::cout << "all_proved =" << !res2 << std::endl;
    std::cout << "queries = " << queries << std::endl;
    return 0;
}