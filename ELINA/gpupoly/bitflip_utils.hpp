#ifndef __GPUPOLY_BITFLIP_UTILS_HPP__
#define __GPUPOLY_BITFLIP_UTILS_HPP__

#include <cassert>
#include <iostream>
#include <vector>

// Flip the k-th bit of the value. Use bit_all-bit two's complement form. return the correct flipped value
int flip_bit(int value, int k_th, int bit_all);

// Flip the masked bit of the value. Use bit_all-bit two's complement form. return the correct flipped value
int flip_bit_mask(int value, int mask, int bit_all);

// Get all possible bit mask which has at most cnt's 1s.
std::vector<int> getBitMask(int bit_all, int cnt);

// Get all sample points in order with the given masks
std::vector<double> getFlippedWeightMasks(int value, int bit_all, const std::vector<int>& masks, double DeltaW);

// Flip cnt bits of the value. Use bit_all-bit two's complement form. return the correct flipped value. return the value's range
std::pair<int, int> rangeFlipKBitInt(int value, int bit_all, int cnt, int mask = 0);

// Use bit_all-bit two's complement form. return the correct flipped value. using the masks
std::pair<int, int> rangeFlipKBitIntMasks(int value, int bit_all, const std::vector<int>& masks);

// Flip cnt bits of the value except for the sign bit. Use bit_all-bit two's complement form. return the correct flipped value. return the value's range
std::pair<int, int> rangeFlipKBitIntPreserve(int value, int bit_all, int cnt, int mask = 0);

// The vulunrable parameters
struct target
{
        int layerIndex;
        int neuronIndex;
        int weightIndex;
};


// The activation type
enum ActivationType
{
        ReLU,
        Sigmoid,
        Tanh,
};

// the neuralnetwork information
struct nnInfo
{
        std::vector<std::vector<std::vector<double>>> weights;
        std::vector<std::vector<double>> bias;
        std::vector<int> layerSizes;
        std::vector<double> DeltaWs;
        std::vector<double> input_lower;
        std::vector<double> input_upper;
        int input_id;
        int label;
        int bit_all;
        ActivationType actType;
};

//Method Enum
enum Method{
    BASELINE, BINARYSEARCH, BFA_RA_WO_BINARY, BFA_ONE_INTERVAL, INTEGRATE_TEST
};

//generate random targets (vulunrable parameters)
std::vector<target> getTargets(int layerIndex, int layerSize, int lastLayerSize, int targets_num);

//read the network from the json file
int readJson(std::string path, nnInfo &info);

//set the input from the json file
int setInputFromPara(std::string path, double radius, nnInfo& info);

//add a variable weight to the netork
int add_variable_weight(NeuralNetwork *nn, int layerIndex, int neuronIndex, int weightIndex, double rangeMin, double rangeMax, int num_virtual_neuron);

//add a varaible bias to the network
int add_variable_bias(NeuralNetwork *nn, int layerIndex, int neuronIndex, double rangeMin, double rangeMax);

//add a relu-skip info
int add_relu_skip(NeuralNetwork *nn, int layerIndex, int neuronIndex, int num_virtual_neuron);

NeuralNetwork *createNN(ActivationType actType,const std::vector<int> &layerSizes, const std::vector<std::vector<std::vector<double>>> &weights, const std::vector<std::vector<double>> &bias, int virtualNeuronNum, size_t &trueLayerSize);

int resetInfo(NeuralNetwork *nn, std::vector<int> layerIndexes, std::vector<int> neuronIndexes, std::vector<int> weightIndexes, std::vector<double> originVals, int num_virtual_bit);

int getTrueAffineIndex(NeuralNetwork *nn, int affineIndex, size_t trueLayerSize);

int printRecorededBounds(NeuralNetwork *nn, int affineLayerIndex, int trueSize, const char* prelude);

int printIntermediateBounds(NeuralNetwork *nn, int affineLayerIndex, int trueSize, const char* prelude);

int printPostactivationBounds(NeuralNetwork *nn, int affineLayerIndex, int trueSize, const char* prelude);

#endif