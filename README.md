# BFAVerifier

BFAVerifier implementation contains two parts: BFAVerifier and GPUPoly.

BFAVerifier is the algorithm designed in our paper, and it contains BFA\_RA and BFA\_MILP which is easy to read and understand. And GPUPoly is an accelerated version of BFA\_RA by CUDA, harnessing the power of GPU. 

# Installation

## Untar the data set

```
7z x mnistinput.7z
cd ./BFAVerifier
7z x benchmark.7z
```

## Install Python BFAVerifier

BFAVerifier contains BFA\_MILP and a prototype of BFA\_RA for easier understanding.

Please properly setup (and activate) Gurobi and install the Gurobi python package, export the necessary environment variables. Before running the code, please make sure the Gurobi license is properly set up.

The following is the installation of the python environment, Gurobi setup is not included.

```bash
cd BFAVerifier
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you are having issues with the installation of `requirements.txt`, please switch to `Python 3.10.12`, or remove all 
version string in `requirements.txt` and try again.

## Install GPU BFA\_RA Sympoly

GPUPoly contains an accelerated version of BFA\_RA by harnessing the power of GPU, which is based on [ELINA](https://github.com/eth-sri/ELINA/tree/master/gpupoly). 

Please install cuda amd make `nvcc` available in the PATH.

```bash
cd ELINA
./configure -use-cuda
cd GPUPoly
cmake .
make -j 
```

And you shall see `testSympoly` which is the executable for the GPU version of BFA\_RA.

Also, please export `$LD_LIBRARY_PATH` to the cuda library path. This does not require manual operation very often, but if you encountered problems later, make sure `libcublas.so.11` (perhaps, `libcublas.so`) is under the path(s) specified by the `LD_LIBRARY_PATH` (to be included in the runtime environment).

You don't often require to do this, if you encounter related issues. First make sure the dynamic link library is correctly set. `libcublas.so.11` (perhaps `libcublas.so`) is required.

```bash
export LD_LIBRARY_PATH=/path/to/cuda/targets/x86_64-linux/lib
```
Note that you should modify the `LD_LIBRARY_PATH` to the correct path of `libcublas.so.11` (or `libcublas.so`).

# Usage

### Test BFA\_RA GPU Accelerated Version

Try on an MNIST test.

```bash
./testSympoly ../../mnist_weight_other_ptq/GPU_PTQ_.4.0.5blk_100_100_100_100_100.432.json ../../mnistinput/1461.json binarysearch_all -1 2 0.00784313725490196 record_union_bounds 
```

Try on an ACASXU test.

```bash
./testGPUPoly ../../BFAVerifier/benchmark/acasxu/GPU_ACASXUPTQ_.8.-1.nnet_5_2.prop3*4.json binarysearch_all -1 1
```

Detailed usage:

```
<weight file> <input json> <method (binarysearch_all, baseline, relu_skip, bfa_ra_wo_binary, bfa_ra_one_interval, integrate_test)> <targets_per_layer> <bit_flip_cnt> <eps_float> <record_union_bounds (optional)>
```


- `<weight file>` is the weight file in json, which are stored at `../mnist_weight_ptq`, `../mnist_weight_other_activations` and `../BFAVerifier/benchmark/acasxu` for the experiments.
- `<input file>` is the mnist input json file, which contains the 1-normalized input and its ground truth label.
- `<method>` is the method to use, which can be `binarysearch_all`, `baseline`, (`relu_skip`), `bfa_ra_wo_binary`,( `bfa_ra_one_interval`), `integrate_test`. The bracketed methods are for debugging purposes.
- `<targets_per_layer>` is the number of sampling targets per layer, `-1` means all targets.
- `<bit_flip_cnt>` is the number of bit-flip count should not be zero.
- `<eps_float>` is the epsilon float, which is the epsilon for the interval. Typical values: `0.00392156862745098` `0.00784313725490196` and `0.01568627450980392` (1/255, 2/255, 4/255).
- `<record_union_bounds>` is optional, if specified, the union bounds will be recorded and print to the stdout. It's useful to the BFA\_MILP, but not required.


or 

```
<json file> <method (binarysearch_all, baseline, relu_skip, bfa_ra_wo_binary, bfa_ra_one_interval, integrate_test)> <targets_per_layer> <bit_flip_cnt> <record_union_bounds (optional)>
```

for acas xu, the input information is encoded in the json file, so it does not require a `<input json>`.

### Generation of the weight json file

The weight json files are already provided in the `mnist_weight_ptq`, `mnist_weight_other_activations` and `benchmark/acasxu` folders.

For MNIST test, the user should use the following command to generate the input json for GPUPoly.

```bash
python test_SymPoly.py --bit_all 4 --PTQ 1 --arch 3blk_10_10_10  --method baseline --sample_id 5 --targets_per_layer 1 --description randomtargets --bit_only_signed 1 --also_qu_bias 1 --save_test_path "../ELINA/GPUPoly/info.json"
```

For ACASXU test, the json file (along with property info) is already provided in the `benchmark/acasxu` folder.

```bash
BFAVerifier/benchmark/acasxu
```

GPU_QAT_.4.4.5blk_100_100_100_100_100.1461.CNT2.TAR-1.json.res.milp

### Test BFA\_MILP

Before running BFA\_MILP, it's necessary to make GPU BFA\_RA run first to generate the necessary information for BFA\_MILP.

And make sure the Gurobi license and environment are all properly set up.

To run BFA\_MILP, BFA\_MILP accept a `--res_file` which is the stdout of testSympoly. BFA\_MILP automatically reads information from the stdout of testSympoly and setup MILP task then perform the MILP task.

You should redirect the stdout of testSympoly to a file, and abide by the following command.

```
cd ELINA/gpupoly
./testSympoly ../../mnist_weight_ptq/GPU_PTQ_.4.0.5blk_100_100_100_100_100.432.json ../../mnistinput/1461.json binarysearch_all -1 2 0.00784313725490196 record_union_bounds > /tmp/GPU_PTQ_.4.4.5blk_100_100_100_100_100.1461.CNT2.TAR-1.json.res
```

```bash
cd BFAVerifier
python test_milp_mnist.py --res_file /tmp/GPU_PTQ_.4.4.5blk_100_100_100_100_100.1461.CNT2.TAR-1.json.res --hint_file /tmp/GPU_PTQ_.4.4.5blk_100_100_100_100_100.1461.CNT2.TAR-1.json.res
```

For ACAS XU test, use another file but the usage is the same.

```bash
python test_milp_acas.py --res_file GPU_ACASXUPTQaCNT1_.8.-1.nnet_1_5.prop10*0.json.res --hint_file GPU_ACASXUPTQaCNT1_.8.-1.nnet_1_5.prop10*0.json.res
```

Run BFA\_RA. 

```bash
python test_SymPoly.py --bit_all 4 --PTQ 1 --arch 3blk_10_10_10  --method baseline --sample_id 5 --targets_per_layer 1 --description randomtargets --bit_only_signed 1 --also_qu_bias 1
```

### Trouble Shooting

`./testSympoly: error while loading shared libraries: libcublas.so.11: cannot open shared object file: No such file or directory`

This means `libcublas.so.11` is not found in `LD_LIBRARY_PATH`, please find the path to `libcublas.so.11` and export to `LD_LIBRARY_PATH`.
