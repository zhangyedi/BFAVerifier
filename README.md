# BFAVerifier

This is the official webpage for the paper "Verification of Bit-Flip Attacks against Quantized Neural Networks". In this paper, we make the following main contributions:

- We propose a novel abstract domain DeepPolyR to conduct reachability analysis for neural networks with symbolic parameters soundly and efficiently;
- We introduced the first sound, complete, and reasonably efficient bit-flip attacks verification method BFAVerifier for QNNs by cleverly combining deepPolyR and an MILP-based method;
- We implement BFAVerifier as an end-to-end tool and conduct an extensive evaluation of various verification tasks, demonstrating its effectiveness and efficiency.

## Setup

Todo.

## Running BFAVerifier on the benchmarks

Todo.
```
# This is for comment information
python main.py --dataset mnist --arch 3blk_50 --sample_id 100 --r 2 --all_bit 8 --flip_bit 2

```
