# BFAVerifier

BFAVerifier implementation contains two parts: BFAVerifier and GPUSymPoly.

BFAVerifier is the algorithm designed in our paper, and it contains BFA\_RA and BFA\_MILP. And GPUSymPoly is an accelerated version of BFA\_RA by harnessing the power of GPU.

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

If you are having issues with the installation of `requirements.txt`, please switch to `Python 3.10.12`, or remove all 
version string in `requirements.txt` and try again.

## GPUSymPoly

GPUSymPoly contains an accelerated version of BFA\_RA by harnessing the power of GPU, which is based on [ELINA](https://github.com/eth-sri/ELINA/tree/master/gpupoly). 

Please install cuda amd make `nvcc` available in the PATH.

Also, please export `$LD_LIBRARY_PATH` to the cuda library path. `libcublas.so.11` (perhaps, `libcublas.so`) is required to be under the path(s) specified by the `LD_LIBRARY_PATH` (to be included in the runtime environment)

Out-of-the box installation is under development. A pre-built binary executable of testGPUPoly is available. 

```bash
cd ELINA
./configure -use-cuda
cd gpusympoly
cmake .
make -j 
```

And you shall see `testGPUSymPoly` which is the executable for the GPU version of SymPoly.

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
python test_MILP.py --qu_bit 4 --flip_bit 1 --rad 0 --arch 5blk_100_100_100_100_100 --sample_id 432 --parameters_file ./GPU_QAT_.4.0.5blk_100_100_100_100_100.432.CNT1.TAR-1.json.res.parameters
```

```bash
python test_MILP_acas.py --parameters_file GPU_ACASXUPTQaCNT2_.8.-1.nnet_3_7.prop3*2.json.res.parameters --weightPath benchmark/acasxu_h5/GPU_ACASXUPTQaCNT2_.8.-1.nnet_3_7.prop3*2.h5 --instance_file benchmark/acasxu/GPU_ACASXUPTQaCNT2_.8.-1.nnet_3_7.prop3*2.json
```

The `.parameters` file contains unknown parameters generated by BFA\_RA. Use the output of `testGPUSymPoly` and `*_prove_parameter_parser.py` to generate the `.parameters` file, or manually create one.

Run BFA\_RA. 

```bash
python test_SymPoly.py --bit_all 4 --QAT 1 --arch 3blk_10_10_10  --method baseline --sample_id 5 --targets_per_layer 1 --description randomtargets --bit_only_signed 1 --also_qu_bias 1
```

However, we recommend GPUSymPoly for better performance.

### Test GPUSymPoly

First make sure the dynamic link library is correctly set. `libcublas.so.11` (perhaps `libcublas.so`) is required.

```bash
export LD_LIBRARY_PATH=/path/to/cuda/targets/x86_64-linux/lib
```

Note that you should modify the `LD_LIBRARY_PATH` to the correct path of `libcublas.so.11` (or `libcublas.so`).

Try on an MNIST test.

```bash
./testGPUSymPoly ./GPU_QAT_.4.0.3blk_10_10_10.432.json binarysearch_all 100 1
```

100 targets per layer, bit-flip count 1.

Try on an ACASXU test.

```bash
./testGPUSymPoly ../../BFAVerifier/benchmark/acasxu/GPU_ACASXUPTQ_.8.-1.nnet_5_2.prop3\*4.json binarysearch_all -1 1
```

### Other Input Files for GPUSymPoly

For MNIST test, the user should use the following command to generate the input json for GPUSymPoly.

```bash
python test_SymPoly.py --bit_all 4 --QAT 1 --arch 3blk_10_10_10  --method baseline --sample_id 5 --targets_per_layer 1 --description randomtargets --bit_only_signed 1 --also_qu_bias 1 --save_test_path "../ELINA/gpusympoly/info.json"
```

For ACASXU test, the json file (along with property info) is already provided in the `benchmark/acasxu` folder.

```bash
BFAVerifier/benchmark/acasxu
```

### Trouble Shooting

`./testGPUSymPoly: error while loading shared libraries: libcublas.so.11: cannot open shared object file: No such file or directory`

This means `libcublas.so.11` is not found in `LD_LIBRARY_PATH`, please find the path to `libcublas.so.11` and export to `LD_LIBRARY_PATH`.
