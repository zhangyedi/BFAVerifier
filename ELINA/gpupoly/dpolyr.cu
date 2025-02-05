#include "dpolyr.h"

GPUMem<false> DPolyRInfoPool::gpuMem;

DPolyRInfoPool::DPolyRInfoPool() : data_(nullptr), n_(0), capacity(0) {}

DPolyRInfoPool::DPolyRInfoPool(size_t n) : n_(n), capacity(memSize())
{
    if (n_ > 0)
        data_ = static_cast<DPolyRInfo *>(gpuMem.alloc(capacity));
    else
        data_ = nullptr;
}

DPolyRInfoPool::DPolyRInfoPool(const std::vector<DPolyRInfo> &polyrinfo) : DPolyRInfoPool(polyrinfo.size())
{
    if (n_ > 0)
        gpuErrchk(cudaMemcpy(data_, polyrinfo.data(), memSize(), cudaMemcpyHostToDevice));
}

DPolyRInfoPool::DPolyRInfoPool(const DPolyRInfoPool &other) : DPolyRInfoPool(other.size())
{
    if (n_ > 0)
        gpuErrchk(cudaMemcpy(data_, other.data_, memSize(), cudaMemcpyDeviceToDevice));
}

DPolyRInfoPool &DPolyRInfoPool::DPolyRInfoPool::operator=(const DPolyRInfoPool &other)
{
    if (n_ != other.size())
    {
        if (data_ != nullptr)
            gpuMem.free(data_, memSize());
        n_ = other.size();
        capacity = memSize();
        data_ = static_cast<DPolyRInfo *>(gpuMem.alloc(capacity));
    }
    if (n_ > 0)
        gpuErrchk(cudaMemcpy(data_, other.data_, memSize(), cudaMemcpyDeviceToDevice));
    return *this;
}

DPolyRInfoPool& DPolyRInfoPool::operator=(const std::vector<DPolyRInfo> &polyrinfo)
{
    if (n_ != polyrinfo.size())
    {
        if (data_ != nullptr)
            gpuMem.free(data_, memSize());
        n_ = polyrinfo.size();
        capacity = memSize();
        data_ = static_cast<DPolyRInfo *>(gpuMem.alloc(capacity));
    }
    if (n_ > 0)
        gpuErrchk(cudaMemcpy(data_, polyrinfo.data(), memSize(), cudaMemcpyHostToDevice));
    return *this;
}

void DPolyRInfoPool::addInfo(const DPolyRInfo &info)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if(memSize() >= capacity)
    {
        //allocate new memory. double it's capcacity. move old contents. free old resource
        size_t newCapacity = capacity == 0 ? sizeof(DPolyRInfo) : 2 * capacity;
        DPolyRInfo *newData = static_cast<DPolyRInfo *>(gpuMem.alloc(newCapacity));
        assert(newData != nullptr);
        if (data_ != nullptr)
        {
            gpuErrchk(cudaMemcpy(newData, data_, memSize(), cudaMemcpyDeviceToDevice));
            gpuMem.free(data_, capacity);
        }
        data_ = newData;
        capacity = newCapacity;
    }
    // info.print();
    //print info    
    gpuErrchk(cudaMemcpy(data_ + n_, &info, sizeof(DPolyRInfo), cudaMemcpyHostToDevice));
    ++n_;
}

void DPolyRInfoPool::clear(){
    n_ = 0;
}