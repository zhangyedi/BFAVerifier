# BFAVerifier

BFAVerifier implementation contains two parts: SymPoly and GPUSymPolyR.

Out-of-the-box kit is under development.

# Installation

## BFAVerifier

BFAVerifier contains BFA\_MILP and a prototype of BFA\_RA for easier understanding.

Please properly setup (and activate) Gurobi and install the Gurobi python package, export the necessary environment variables. 


```bash
cd BFAVerifier
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## GPUSymPoly

GPUSymPoly contains an accelerated version of BFA\_RA by harnessing the power of GPU, which is based on [ELINA](https://github.com/eth-sri/ELINA/tree/master/gpupoly). 

Please install cuda amd make nvcc available in the PATH.

Also, please export `$LD_LIBRARY_PATH` to the cuda library path. libcublas.so.11 is required to be under the path(s) specified by the `LD_LIBRARY_PATH`

Out-of-the box installation is under development. A pre-built binary executable of testGPUPoly is available. 

```bash
cd ELINA
./configure -use-deeppoly -use-fconv -use-cuda
cd gpupoly
cmake .
make -j 
```

And you shall see `testGPUPoly` which is the executable for the GPU version of SymPoly.

# Usage

## BFAVerifier

### Test BFAVerifier

Execute the following command to test your environment setup.

```bash
cd BFAVerifier
source venv/bin/activate
python validate.py
```

Run BFA\_MILP

```bash
python test_MILP.py --qu_bit 4 --flip_bit 1 --rad 0 --arch 5blk_100_100_100_100_100 --sample_id 432 --parameters_f
ile ./GPU_QAT_.4.0.5blk_100_100_100_100_100.432.CNT1.TAR-1.json.res.parameters
```

Run BFA\_RA

```bash
python test_SymPoly.py --bit_all 4 --QAT 1 --arch 3blk_10_10_10  --method baseline --sample_id 5 --targets_per_la
yer 1 --description randomtargets --bit_only_signed 1 --also_qu_bias 1
```

Generate input for GPUSymPoly

```bash
python GPUSymPoly.py --bit_all 4 --QAT 1 --arch 3blk_10_10_10  --method baseline --sample_id 5 --targets_per_layer 1 --description randomtargets --bit_only_signed 1 --also_qu_bias 1 --save_test_path "../ELINA/gpupoly/info.json"
```

### Test GPUSymPoly

First make sure the dynamic link library is correctly set. `libcublas.so.11` is required.

```bash
export LD_LIBRARY_PATH=/path/to/cuda/targets/x86_64-linux/lib
```

```bash
./testGPUSymPoly ./GPU_QAT_.4.0.3blk_10_10_10.432.json binarysearch_all 100 1
```

100 targets per layer, bit-flip count 1.

### Trouble Shooting

`./testGPUSymPoly: error while loading shared libraries: libcublas.so.11: cannot open shared object file: No such file or directory`

This means `libcublas.so.11` is not found in `LD_LIBRARY_PATH`, please find the path to `libcublas.so.11` and export to `LD_LIBRARY_PATH`.
