#pragma once
#include <cassert>
#include <vector>
#include <iostream>
#include "gpumem.h"
#include <mutex>

struct DPolyRInfo{
    enum class AttackType{ NOATTACK, BITFLIP_RANGE_ABSTRACT, GLICTHCING };
    /// @brief The attack type
    AttackType attackType;
    /// @brief The range of the bitflip attack
	double rangeMin,rangeMax;
    /// @brief The neuron index of the previous layer
    int preNeuronIndex;
    /// @brief The neuron index of the current virtual neuron
    int curVirtualNeuronIndex;
    void print()const{
        std::cout << "AttackType: ";
        const char *attackTypeStr[] = {"NOATTACK", "BITFLIP_RANGE_ABSTRACT", "GLICTHCING"};
        std::cout << attackTypeStr[(int)attackType] << std::endl;
        std::cout << "rangeMin: " << rangeMin << std::endl;
        std::cout << "rangeMax: " << rangeMax << std::endl;
        std::cout << "preNeuronIndex: " << preNeuronIndex << std::endl;
        std::cout << "curVirtualNeuronIndex: " << curVirtualNeuronIndex << std::endl;
    }
    
};

class DPolyRInfoPool{
public:
    static GPUMem<false> gpuMem; 
private:
    /// @brief pointer to the data in the GPU memory
    DPolyRInfo *data_;

    /// @brief number of elements in the pool
    size_t n_;

    /// @brief capacity of the pool in bytes
    size_t capacity;

    /// @brief lock
    std::mutex mutex_;
public:
    DPolyRInfoPool ();
    DPolyRInfoPool (size_t);
    DPolyRInfoPool (const std::vector<DPolyRInfo> &); // convert from std::vector from host
    
    size_t memSize() const { return n_ * sizeof(DPolyRInfo); }
    size_t size() const {return n_;}
    
    DPolyRInfoPool(const DPolyRInfoPool&); // copy constructor
    DPolyRInfoPool& operator=(const DPolyRInfoPool&); // copy assignment
    DPolyRInfoPool& operator=(const std::vector<DPolyRInfo>&); // copy assignment from std::vector
    ~DPolyRInfoPool (){ // destructor
        if(data_ != nullptr){
            gpuMem.free(data_, capacity);
        }
    }

    const DPolyRInfo * data() const { return data_; }
    void addInfo(const DPolyRInfo&);
    void print()const{
        DPolyRInfo* data = new DPolyRInfo[n_];
        cudaMemcpy(data, data_, n_ * sizeof(DPolyRInfo), cudaMemcpyDeviceToHost);
        for(size_t i = 0; i < n_; i++){
            data[i].print();
        }
        delete[] data;
    }
    void clear();
};