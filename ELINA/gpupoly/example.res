GPUPoly 0.13.0S Debug (built Apr 26 2024 13:58:03) - Copyright (C) 2020 Department of Computer Science, ETH Zurich.
This program comes with ABSOLUTELY NO WARRANTY. This is free software, and you are welcome to redistribute it and to modify it under the terms of the GNU LGPLv3.

Warning: bit-flip at the first layer is not supported due to heavy technical debt (CPU version is supported)
Validating Bias
weights_path: benchmark/benchmark_QAT/QAT_mnist_3blk_10_10_10_qu_4.h5
====
Appending input layer
info.layerSizes.size() = 4
info.layerSizes.size() = 5
info.weights.size() = 4
info.weights.size() = 5
info.bias.size() = 4
info.bias.size() = 5
weights.size()'s are (1,0) (10,784) (10,10) (10,10) (10,10) 
bias.size()'s are 1 10 10 10 10 
DeltaWs.size() is 5
input_lower.size() is 784
input_upper.size() is 784
label is 1
bit_all is 4
layerSizes are
784 10 10 10 10 
json file: ./info.json
method: binarysearch_bias
targets_per_layer: 100
bit_flip_cnt: 1

!DEBUG: Infer once with no flips
no_flip_proved=1
!DEBUG: Infer once with no flips Ended
Warning: layer 2 has less than 100 targets
Warning: layer 3 has less than 100 targets
Warning: layer 4 has less than 100 targets
preprocessing end
(Divide and Counter) Proved 2 0 (bias) by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 0 (bias) by DeepPolyR at [0, 4]=[-2.134562, -0.355760]
(DeepPolyR) Proved 2 0 (bias) by DeepPolyR at [4, 4]=[2.134562, 2.134562]
(Overall) Proved 2 0 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 1 (bias) by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 1 (bias) by DeepPolyR at [0, 4]=[-2.490323, -2.490323]
(DeepPolyR) Proved 2 1 (bias) by DeepPolyR at [1, 4]=[0.000000, 1.778802]
(Overall) Proved 2 1 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 (bias) by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 2 (bias) by DeepPolyR at [0, 4]=[-2.846083, -2.846083]
(DeepPolyR) Proved 2 2 (bias) by DeepPolyR at [1, 4]=[0.000000, 1.423042]
(Overall) Proved 2 2 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 (bias) by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 3 (bias) by DeepPolyR at [0, 4]=[-2.134562, -0.355760]
(DeepPolyR) Proved 2 3 (bias) by DeepPolyR at [4, 4]=[2.134562, 2.134562]
(Overall) Proved 2 3 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 (bias) by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 4 (bias) by DeepPolyR at [0, 4]=[-2.134562, -0.355760]
(DeepPolyR) Proved 2 4 (bias) by DeepPolyR at [4, 4]=[2.134562, 2.134562]
(Overall) Proved 2 4 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 (bias) by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 5 (bias) by DeepPolyR at [0, 4]=[-2.846083, -2.846083]
(DeepPolyR) Proved 2 5 (bias) by DeepPolyR at [1, 4]=[0.000000, 1.423042]
(Overall) Proved 2 5 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 (bias) by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 6 (bias) by DeepPolyR at [0, 4]=[-2.846083, -2.846083]
(DeepPolyR) Proved 2 6 (bias) by DeepPolyR at [1, 4]=[0.000000, 1.423042]
(Overall) Proved 2 6 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 (bias) by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 7 (bias) by DeepPolyR at [0, 4]=[-2.134562, 0.000000]
(DeepPolyR) Proved 2 7 (bias) by DeepPolyR at [2, 4]=[0.711521, 2.134562]
(Overall) Proved 2 7 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 (bias) by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 8 (bias) by DeepPolyR at [0, 4]=[-2.134562, -0.355760]
(DeepPolyR) Proved 2 8 (bias) by DeepPolyR at [4, 4]=[2.134562, 2.134562]
(Overall) Proved 2 8 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 (bias) by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 9 (bias) by DeepPolyR at [0, 4]=[-2.490323, -2.490323]
(DeepPolyR) Proved 2 9 (bias) by DeepPolyR at [1, 4]=[0.000000, 1.778802]
(Overall) Proved 2 9 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 (bias) by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 0 (bias) by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 0 (bias) by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 0 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 (bias) by DeepPolyR at [0, 4]=[-2.427669, 1.040430]
(DeepPolyR) Proved 3 1 (bias) by DeepPolyR at [0, 4]=[-2.427669, -1.734049]
(DeepPolyR) Proved 3 1 (bias) by DeepPolyR at [3, 4]=[-0.346810, 1.040430]
(Overall) Proved 3 1 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 2 (bias) by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 2 (bias) by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 2 (bias) by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 2 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 (bias) by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 3 (bias) by DeepPolyR at [0, 4]=[-2.080859, -0.346810]
(DeepPolyR) Proved 3 3 (bias) by DeepPolyR at [4, 4]=[2.080859, 2.080859]
(Overall) Proved 3 3 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 (bias) by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 4 (bias) by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 4 (bias) by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 4 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 (bias) by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 5 (bias) by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 5 (bias) by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 5 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 (bias) by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 6 (bias) by DeepPolyR at [0, 4]=[-2.427669, -0.346810]
(DeepPolyR) Proved 3 6 (bias) by DeepPolyR at [4, 4]=[1.734049, 1.734049]
(Overall) Proved 3 6 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 7 (bias) by DeepPolyR at [0, 4]=[-2.774479, 1.387240]
(DeepPolyR) Proved 3 7 (bias) by DeepPolyR at [0, 4]=[-2.774479, -2.774479]
(DeepPolyR) Proved 3 7 (bias) by DeepPolyR at [1, 4]=[0.000000, 1.387240]
(Overall) Proved 3 7 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 8 (bias) by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 8 (bias) by DeepPolyR at [0, 4]=[-1.734049, -1.734049]
(DeepPolyR) Proved 3 8 (bias) by DeepPolyR at [1, 4]=[0.346810, 2.427669]
(Overall) Proved 3 8 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 (bias) by DeepPolyR at [0, 4]=[-2.774479, 1.387240]
(DeepPolyR) Proved 3 9 (bias) by DeepPolyR at [0, 4]=[-2.774479, -2.774479]
(DeepPolyR) Proved 3 9 (bias) by DeepPolyR at [1, 4]=[0.000000, 1.387240]
(Overall) Proved 3 9 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 (bias) by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 0 (bias) by DeepPolyR at [0, 4]=[-4.211204, -4.211204]
(DeepPolyR) Proved 4 0 (bias) by DeepPolyR at [1, 4]=[0.000000, 3.008003]
(Overall) Proved 4 0 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 1 (bias) by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 1 (bias) by DeepPolyR at [0, 4]=[-3.609603, -0.601601]
(DeepPolyR) Proved 4 1 (bias) by DeepPolyR at [4, 4]=[3.609603, 3.609603]
(Overall) Proved 4 1 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 (bias) by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 2 (bias) by DeepPolyR at [0, 4]=[-4.812805, -4.812805]
(DeepPolyR) Proved 4 2 (bias) by DeepPolyR at [1, 4]=[0.000000, 2.406402]
(Overall) Proved 4 2 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 (bias) by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 3 (bias) by DeepPolyR at [0, 4]=[-4.812805, -4.812805]
(DeepPolyR) Proved 4 3 (bias) by DeepPolyR at [1, 4]=[0.000000, 2.406402]
(Overall) Proved 4 3 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 4 (bias) by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 4 (bias) by DeepPolyR at [0, 4]=[-3.008003, 0.601601]
(DeepPolyR) Proved 4 4 (bias) by DeepPolyR at [2, 4]=[1.203201, 4.211204]
(Overall) Proved 4 4 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 (bias) by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 5 (bias) by DeepPolyR at [0, 4]=[-3.008003, -0.601601]
(DeepPolyR) Proved 4 5 (bias) by DeepPolyR at [4, 4]=[4.211204, 4.211204]
(Overall) Proved 4 5 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 (bias) by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 6 (bias) by DeepPolyR at [0, 4]=[-3.008003, -0.601601]
(DeepPolyR) Proved 4 6 (bias) by DeepPolyR at [4, 4]=[4.211204, 4.211204]
(Overall) Proved 4 6 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 7 (bias) by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 7 (bias) by DeepPolyR at [0, 4]=[-4.211204, -4.211204]
(DeepPolyR) Proved 4 7 (bias) by DeepPolyR at [1, 4]=[0.000000, 3.008003]
(Overall) Proved 4 7 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 (bias) by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 8 (bias) by DeepPolyR at [0, 4]=[-4.211204, -4.211204]
(DeepPolyR) Proved 4 8 (bias) by DeepPolyR at [1, 4]=[0.000000, 3.008003]
(Overall) Proved 4 8 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 9 (bias) by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 9 (bias) by DeepPolyR at [0, 4]=[-3.609603, 0.000000]
(DeepPolyR) Proved 4 9 (bias) by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Overall) Proved 4 9 (bias) with all masks. Summary: 1 1 1 

Elapsed Time: 0.268
flg =0
all_proved =1
queries = 30
GPUPoly 0.13.0S Debug (built Apr 26 2024 13:58:03) - Copyright (C) 2020 Department of Computer Science, ETH Zurich.
This program comes with ABSOLUTELY NO WARRANTY. This is free software, and you are welcome to redistribute it and to modify it under the terms of the GNU LGPLv3.

Warning: bit-flip at the first layer is not supported due to heavy technical debt (CPU version is supported)
Validating Bias
weights_path: benchmark/benchmark_QAT/QAT_mnist_3blk_10_10_10_qu_4.h5
====
Appending input layer
info.layerSizes.size() = 4
info.layerSizes.size() = 5
info.weights.size() = 4
info.weights.size() = 5
info.bias.size() = 4
info.bias.size() = 5
weights.size()'s are (1,0) (10,784) (10,10) (10,10) (10,10) 
bias.size()'s are 1 10 10 10 10 
DeltaWs.size() is 5
input_lower.size() is 784
input_upper.size() is 784
label is 1
bit_all is 4
layerSizes are
784 10 10 10 10 
json file: ./info.json
method: binarysearch
targets_per_layer: 100
bit_flip_cnt: 1

!DEBUG: Infer once with no flips
no_flip_proved=1
!DEBUG: Infer once with no flips Ended
preprocessing end
(Divide and Counter) Proved 2 0 0 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 0 0 by DeepPolyR at [0, 4]=[-1.778802, 0.355760]
(DeepPolyR) Proved 2 0 0 by DeepPolyR at [2, 4]=[0.711521, 2.490323]
(Overall) Proved 2 0 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 0 1 by DeepPolyR at [0, 4]=[-0.355760, 2.490323]
(DeepPolyR) Proved 2 0 1 by DeepPolyR at [0, 4]=[-0.355760, 1.067281]
(DeepPolyR) Proved 2 0 1 by DeepPolyR at [2, 4]=[1.778802, 2.490323]
(Overall) Proved 2 0 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 0 2 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 0 2 by DeepPolyR at [0, 4]=[-2.490323, -2.490323]
(DeepPolyR) Proved 2 0 2 by DeepPolyR at [1, 4]=[0.000000, 1.778802]
(Overall) Proved 2 0 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 0 3 by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 0 3 by DeepPolyR at [0, 4]=[-2.846083, -2.846083]
(DeepPolyR) Proved 2 0 3 by DeepPolyR at [1, 4]=[0.000000, 1.423042]
(Overall) Proved 2 0 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 0 4 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 0 4 by DeepPolyR at [0, 4]=[-2.490323, -2.490323]
(DeepPolyR) Proved 2 0 4 by DeepPolyR at [1, 4]=[0.000000, 1.778802]
(Overall) Proved 2 0 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 0 5 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 0 5 by DeepPolyR at [0, 4]=[-2.134562, -0.355760]
(DeepPolyR) Proved 2 0 5 by DeepPolyR at [4, 4]=[2.134562, 2.134562]
(Overall) Proved 2 0 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 0 6 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 0 6 by DeepPolyR at [0, 4]=[-2.134562, 0.000000]
(DeepPolyR) Proved 2 0 6 by DeepPolyR at [2, 4]=[0.711521, 2.134562]
(Overall) Proved 2 0 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 0 7 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 0 7 by DeepPolyR at [0, 4]=[-2.134562, 0.000000]
(DeepPolyR) Proved 2 0 7 by DeepPolyR at [2, 4]=[0.711521, 2.134562]
(Overall) Proved 2 0 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 0 8 by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 0 8 by DeepPolyR at [0, 4]=[-2.846083, -2.846083]
(DeepPolyR) Proved 2 0 8 by DeepPolyR at [1, 4]=[0.000000, 1.423042]
(Overall) Proved 2 0 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 0 9 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 0 9 by DeepPolyR at [0, 4]=[-2.134562, -0.355760]
(DeepPolyR) Proved 2 0 9 by DeepPolyR at [4, 4]=[2.134562, 2.134562]
(Overall) Proved 2 0 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 1 0 by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 1 0 by DeepPolyR at [0, 4]=[-2.846083, -2.846083]
(DeepPolyR) Proved 2 1 0 by DeepPolyR at [1, 4]=[0.000000, 1.423042]
(Overall) Proved 2 1 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 1 1 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 1 1 by DeepPolyR at [0, 4]=[-2.490323, -2.490323]
(DeepPolyR) Proved 2 1 1 by DeepPolyR at [1, 4]=[0.000000, 1.778802]
(Overall) Proved 2 1 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 1 2 by DeepPolyR at [0, 4]=[-1.423042, 2.134562]
(DeepPolyR) Proved 2 1 2 by DeepPolyR at [0, 4]=[-1.423042, 0.000000]
(DeepPolyR) Proved 2 1 2 by DeepPolyR at [2, 4]=[1.423042, 2.134562]
(Overall) Proved 2 1 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 1 3 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 1 3 by DeepPolyR at [0, 4]=[-1.778802, -0.355760]
(DeepPolyR) Proved 2 1 3 by DeepPolyR at [4, 4]=[2.490323, 2.490323]
(Overall) Proved 2 1 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 1 4 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 1 4 by DeepPolyR at [0, 4]=[-2.134562, 0.000000]
(DeepPolyR) Proved 2 1 4 by DeepPolyR at [2, 4]=[0.711521, 2.134562]
(Overall) Proved 2 1 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 1 5 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 1 5 by DeepPolyR at [0, 4]=[-1.778802, 0.355760]
(DeepPolyR) Proved 2 1 5 by DeepPolyR at [2, 4]=[0.711521, 2.490323]
(Overall) Proved 2 1 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 1 6 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 1 6 by DeepPolyR at [0, 4]=[-2.490323, -1.067281]
(DeepPolyR) Proved 2 1 6 by DeepPolyR at [3, 4]=[-0.355760, 1.778802]
(Overall) Proved 2 1 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 1 7 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 1 7 by DeepPolyR at [0, 4]=[-2.490323, -1.067281]
(DeepPolyR) Proved 2 1 7 by DeepPolyR at [3, 4]=[-0.355760, 1.778802]
(Overall) Proved 2 1 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 1 8 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 1 8 by DeepPolyR at [0, 4]=[-2.490323, -1.067281]
(DeepPolyR) Proved 2 1 8 by DeepPolyR at [3, 4]=[-0.355760, 1.778802]
(Overall) Proved 2 1 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 1 9 by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 1 9 by DeepPolyR at [0, 4]=[-2.846083, -2.846083]
(DeepPolyR) Proved 2 1 9 by DeepPolyR at [1, 4]=[0.000000, 1.423042]
(Overall) Proved 2 1 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 0 by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 2 0 by DeepPolyR at [0, 4]=[-2.846083, -0.711521]
(DeepPolyR) Proved 2 2 0 by DeepPolyR at [4, 4]=[1.423042, 1.423042]
(Overall) Proved 2 2 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 1 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 2 1 by DeepPolyR at [0, 4]=[-2.490323, -1.067281]
(DeepPolyR) Proved 2 2 1 by DeepPolyR at [3, 4]=[-0.355760, 1.778802]
(Overall) Proved 2 2 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 2 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 2 2 by DeepPolyR at [0, 4]=[-2.490323, -2.490323]
(DeepPolyR) Proved 2 2 2 by DeepPolyR at [1, 4]=[0.000000, 1.778802]
(Overall) Proved 2 2 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 3 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 2 3 by DeepPolyR at [0, 4]=[-2.134562, -0.355760]
(DeepPolyR) Proved 2 2 3 by DeepPolyR at [4, 4]=[2.134562, 2.134562]
(Overall) Proved 2 2 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 4 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 2 4 by DeepPolyR at [0, 4]=[-1.778802, 0.355760]
(DeepPolyR) Proved 2 2 4 by DeepPolyR at [2, 4]=[0.711521, 2.490323]
(Overall) Proved 2 2 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 5 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 2 5 by DeepPolyR at [0, 4]=[-1.778802, 0.355760]
(DeepPolyR) Proved 2 2 5 by DeepPolyR at [2, 4]=[0.711521, 2.490323]
(Overall) Proved 2 2 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 6 by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 2 6 by DeepPolyR at [0, 4]=[-2.846083, -2.846083]
(DeepPolyR) Proved 2 2 6 by DeepPolyR at [1, 4]=[0.000000, 1.423042]
(Overall) Proved 2 2 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 7 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 2 7 by DeepPolyR at [0, 4]=[-2.490323, -1.067281]
(DeepPolyR) Proved 2 2 7 by DeepPolyR at [3, 4]=[-0.355760, 1.778802]
(Overall) Proved 2 2 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 8 by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 2 8 by DeepPolyR at [0, 4]=[-2.846083, -2.846083]
(DeepPolyR) Proved 2 2 8 by DeepPolyR at [1, 4]=[0.000000, 1.423042]
(Overall) Proved 2 2 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 9 by DeepPolyR at [0, 4]=[-1.067281, 2.490323]
(DeepPolyR) Proved 2 2 9 by DeepPolyR at [0, 4]=[-1.067281, 0.355760]
(DeepPolyR) Proved 2 2 9 by DeepPolyR at [2, 4]=[1.423042, 2.490323]
(Overall) Proved 2 2 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 0 by DeepPolyR at [0, 4]=[-1.067281, 2.490323]
(DeepPolyR) Proved 2 3 0 by DeepPolyR at [0, 4]=[-1.067281, 0.355760]
(DeepPolyR) Proved 2 3 0 by DeepPolyR at [2, 4]=[1.423042, 2.490323]
(Overall) Proved 2 3 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 1 by DeepPolyR at [0, 4]=[-1.067281, 2.490323]
(DeepPolyR) Proved 2 3 1 by DeepPolyR at [0, 4]=[-1.067281, 0.355760]
(DeepPolyR) Proved 2 3 1 by DeepPolyR at [2, 4]=[1.423042, 2.490323]
(Overall) Proved 2 3 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 2 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 3 2 by DeepPolyR at [0, 4]=[-2.134562, -0.355760]
(DeepPolyR) Proved 2 3 2 by DeepPolyR at [4, 4]=[2.134562, 2.134562]
(Overall) Proved 2 3 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 3 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 3 3 by DeepPolyR at [0, 4]=[-1.778802, 0.355760]
(DeepPolyR) Proved 2 3 3 by DeepPolyR at [2, 4]=[0.711521, 2.490323]
(Overall) Proved 2 3 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 4 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 3 4 by DeepPolyR at [0, 4]=[-1.778802, -0.355760]
(DeepPolyR) Proved 2 3 4 by DeepPolyR at [4, 4]=[2.490323, 2.490323]
(Overall) Proved 2 3 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 5 by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 3 5 by DeepPolyR at [0, 4]=[-2.846083, -0.711521]
(DeepPolyR) Proved 2 3 5 by DeepPolyR at [4, 4]=[1.423042, 1.423042]
(Overall) Proved 2 3 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 6 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 3 6 by DeepPolyR at [0, 4]=[-2.134562, 0.000000]
(DeepPolyR) Proved 2 3 6 by DeepPolyR at [2, 4]=[0.711521, 2.134562]
(Overall) Proved 2 3 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 7 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 3 7 by DeepPolyR at [0, 4]=[-2.134562, 0.000000]
(DeepPolyR) Proved 2 3 7 by DeepPolyR at [2, 4]=[0.711521, 2.134562]
(Overall) Proved 2 3 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 8 by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 3 8 by DeepPolyR at [0, 4]=[-2.846083, -2.846083]
(DeepPolyR) Proved 2 3 8 by DeepPolyR at [1, 4]=[0.000000, 1.423042]
(Overall) Proved 2 3 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 9 by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 3 9 by DeepPolyR at [0, 4]=[-2.846083, -2.846083]
(DeepPolyR) Proved 2 3 9 by DeepPolyR at [1, 4]=[0.000000, 1.423042]
(Overall) Proved 2 3 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 0 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 4 0 by DeepPolyR at [0, 4]=[-2.490323, -1.067281]
(DeepPolyR) Proved 2 4 0 by DeepPolyR at [3, 4]=[-0.355760, 1.778802]
(Overall) Proved 2 4 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 1 by DeepPolyR at [0, 4]=[-2.490323, 1.067281]
(DeepPolyR) Proved 2 4 1 by DeepPolyR at [0, 4]=[-2.490323, -1.778802]
(DeepPolyR) Proved 2 4 1 by DeepPolyR at [3, 4]=[-0.355760, 1.067281]
(Overall) Proved 2 4 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 2 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 4 2 by DeepPolyR at [0, 4]=[-1.778802, 0.355760]
(DeepPolyR) Proved 2 4 2 by DeepPolyR at [2, 4]=[0.711521, 2.490323]
(Overall) Proved 2 4 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 3 by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 4 3 by DeepPolyR at [0, 4]=[-2.846083, -0.711521]
(DeepPolyR) Proved 2 4 3 by DeepPolyR at [4, 4]=[1.423042, 1.423042]
(Overall) Proved 2 4 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 4 by DeepPolyR at [0, 4]=[-1.423042, 2.134562]
(DeepPolyR) Proved 2 4 4 by DeepPolyR at [0, 4]=[-1.423042, 0.000000]
(DeepPolyR) Proved 2 4 4 by DeepPolyR at [2, 4]=[1.423042, 2.134562]
(Overall) Proved 2 4 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 5 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 4 5 by DeepPolyR at [0, 4]=[-1.778802, 0.355760]
(DeepPolyR) Proved 2 4 5 by DeepPolyR at [2, 4]=[0.711521, 2.490323]
(Overall) Proved 2 4 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 6 by DeepPolyR at [0, 4]=[-2.846083, 0.355760]
(DeepPolyR) Proved 2 4 6 by DeepPolyR at [0, 4]=[-2.846083, -1.778802]
(DeepPolyR) Proved 2 4 6 by DeepPolyR at [3, 4]=[-1.067281, 0.355760]
(Overall) Proved 2 4 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 7 by DeepPolyR at [0, 4]=[-2.490323, 1.067281]
(DeepPolyR) Proved 2 4 7 by DeepPolyR at [0, 4]=[-2.490323, -1.778802]
(DeepPolyR) Proved 2 4 7 by DeepPolyR at [3, 4]=[-0.355760, 1.067281]
(Overall) Proved 2 4 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 8 by DeepPolyR at [0, 4]=[-2.846083, 0.711521]
(DeepPolyR) Proved 2 4 8 by DeepPolyR at [0, 4]=[-2.846083, -1.778802]
(DeepPolyR) Proved 2 4 8 by DeepPolyR at [3, 4]=[-0.711521, 0.711521]
(Overall) Proved 2 4 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 9 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 4 9 by DeepPolyR at [0, 4]=[-1.778802, 0.355760]
(DeepPolyR) Proved 2 4 9 by DeepPolyR at [2, 4]=[0.711521, 2.490323]
(Overall) Proved 2 4 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 0 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 5 0 by DeepPolyR at [0, 4]=[-1.778802, -0.355760]
(DeepPolyR) Proved 2 5 0 by DeepPolyR at [4, 4]=[2.490323, 2.490323]
(Overall) Proved 2 5 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 1 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 5 1 by DeepPolyR at [0, 4]=[-2.490323, -1.067281]
(DeepPolyR) Proved 2 5 1 by DeepPolyR at [3, 4]=[-0.355760, 1.778802]
(Overall) Proved 2 5 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 2 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 5 2 by DeepPolyR at [0, 4]=[-2.134562, 0.000000]
(DeepPolyR) Proved 2 5 2 by DeepPolyR at [2, 4]=[0.711521, 2.134562]
(Overall) Proved 2 5 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 3 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 5 3 by DeepPolyR at [0, 4]=[-2.134562, 0.000000]
(DeepPolyR) Proved 2 5 3 by DeepPolyR at [2, 4]=[0.711521, 2.134562]
(Overall) Proved 2 5 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 4 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 5 4 by DeepPolyR at [0, 4]=[-2.134562, -0.355760]
(DeepPolyR) Proved 2 5 4 by DeepPolyR at [4, 4]=[2.134562, 2.134562]
(Overall) Proved 2 5 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 5 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 5 5 by DeepPolyR at [0, 4]=[-2.490323, -2.490323]
(DeepPolyR) Proved 2 5 5 by DeepPolyR at [1, 4]=[0.000000, 1.778802]
(Overall) Proved 2 5 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 6 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 5 6 by DeepPolyR at [0, 4]=[-2.134562, 0.000000]
(DeepPolyR) Proved 2 5 6 by DeepPolyR at [2, 4]=[0.711521, 2.134562]
(Overall) Proved 2 5 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 7 by DeepPolyR at [0, 4]=[-1.067281, 2.490323]
(DeepPolyR) Proved 2 5 7 by DeepPolyR at [0, 4]=[-1.067281, 0.355760]
(DeepPolyR) Proved 2 5 7 by DeepPolyR at [2, 4]=[1.423042, 2.490323]
(Overall) Proved 2 5 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 8 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 5 8 by DeepPolyR at [0, 4]=[-1.778802, 0.355760]
(DeepPolyR) Proved 2 5 8 by DeepPolyR at [2, 4]=[0.711521, 2.490323]
(Overall) Proved 2 5 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 9 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 5 9 by DeepPolyR at [0, 4]=[-2.490323, -2.490323]
(DeepPolyR) Proved 2 5 9 by DeepPolyR at [1, 4]=[0.000000, 1.778802]
(Overall) Proved 2 5 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 0 by DeepPolyR at [0, 4]=[-1.423042, 2.134562]
(DeepPolyR) Proved 2 6 0 by DeepPolyR at [0, 4]=[-1.423042, 0.000000]
(DeepPolyR) Proved 2 6 0 by DeepPolyR at [2, 4]=[1.423042, 2.134562]
(Overall) Proved 2 6 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 1 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 6 1 by DeepPolyR at [0, 4]=[-2.490323, -2.490323]
(DeepPolyR) Proved 2 6 1 by DeepPolyR at [1, 4]=[0.000000, 1.778802]
(Overall) Proved 2 6 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 2 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 6 2 by DeepPolyR at [0, 4]=[-2.490323, -1.067281]
(DeepPolyR) Proved 2 6 2 by DeepPolyR at [3, 4]=[-0.355760, 1.778802]
(Overall) Proved 2 6 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 3 by DeepPolyR at [0, 4]=[-1.423042, 2.134562]
(DeepPolyR) Proved 2 6 3 by DeepPolyR at [0, 4]=[-1.423042, 0.000000]
(DeepPolyR) Proved 2 6 3 by DeepPolyR at [2, 4]=[1.423042, 2.134562]
(Overall) Proved 2 6 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 4 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 6 4 by DeepPolyR at [0, 4]=[-2.490323, -2.490323]
(DeepPolyR) Proved 2 6 4 by DeepPolyR at [1, 4]=[0.000000, 1.778802]
(Overall) Proved 2 6 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 5 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 6 5 by DeepPolyR at [0, 4]=[-1.778802, -0.355760]
(DeepPolyR) Proved 2 6 5 by DeepPolyR at [4, 4]=[2.490323, 2.490323]
(Overall) Proved 2 6 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 6 by DeepPolyR at [0, 4]=[-1.423042, 2.134562]
(DeepPolyR) Proved 2 6 6 by DeepPolyR at [0, 4]=[-1.423042, 0.000000]
(DeepPolyR) Proved 2 6 6 by DeepPolyR at [2, 4]=[1.423042, 2.134562]
(Overall) Proved 2 6 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 7 by DeepPolyR at [0, 4]=[-1.423042, 2.134562]
(DeepPolyR) Proved 2 6 7 by DeepPolyR at [0, 4]=[-1.423042, 0.000000]
(DeepPolyR) Proved 2 6 7 by DeepPolyR at [2, 4]=[1.423042, 2.134562]
(Overall) Proved 2 6 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 8 by DeepPolyR at [0, 4]=[-1.423042, 2.134562]
(DeepPolyR) Proved 2 6 8 by DeepPolyR at [0, 4]=[-1.423042, 0.000000]
(DeepPolyR) Proved 2 6 8 by DeepPolyR at [2, 4]=[1.423042, 2.134562]
(Overall) Proved 2 6 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 9 by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 6 9 by DeepPolyR at [0, 4]=[-2.846083, -2.846083]
(DeepPolyR) Proved 2 6 9 by DeepPolyR at [1, 4]=[0.000000, 1.423042]
(Overall) Proved 2 6 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 0 by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 7 0 by DeepPolyR at [0, 4]=[-2.846083, -2.846083]
(DeepPolyR) Proved 2 7 0 by DeepPolyR at [1, 4]=[0.000000, 1.423042]
(Overall) Proved 2 7 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 1 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 7 1 by DeepPolyR at [0, 4]=[-2.490323, -2.490323]
(DeepPolyR) Proved 2 7 1 by DeepPolyR at [1, 4]=[0.000000, 1.778802]
(Overall) Proved 2 7 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 2 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 7 2 by DeepPolyR at [0, 4]=[-2.134562, -0.355760]
(DeepPolyR) Proved 2 7 2 by DeepPolyR at [4, 4]=[2.134562, 2.134562]
(Overall) Proved 2 7 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 3 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 7 3 by DeepPolyR at [0, 4]=[-1.778802, 0.355760]
(DeepPolyR) Proved 2 7 3 by DeepPolyR at [2, 4]=[0.711521, 2.490323]
(Overall) Proved 2 7 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 4 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 7 4 by DeepPolyR at [0, 4]=[-1.778802, -0.355760]
(DeepPolyR) Proved 2 7 4 by DeepPolyR at [4, 4]=[2.490323, 2.490323]
(Overall) Proved 2 7 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 5 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 7 5 by DeepPolyR at [0, 4]=[-1.778802, -0.355760]
(DeepPolyR) Proved 2 7 5 by DeepPolyR at [4, 4]=[2.490323, 2.490323]
(Overall) Proved 2 7 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 6 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 7 6 by DeepPolyR at [0, 4]=[-2.490323, -2.490323]
(DeepPolyR) Proved 2 7 6 by DeepPolyR at [1, 4]=[0.000000, 1.778802]
(Overall) Proved 2 7 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 7 by DeepPolyR at [0, 4]=[-1.067281, 2.490323]
(DeepPolyR) Proved 2 7 7 by DeepPolyR at [0, 4]=[-1.067281, 0.355760]
(DeepPolyR) Proved 2 7 7 by DeepPolyR at [2, 4]=[1.423042, 2.490323]
(Overall) Proved 2 7 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 8 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 7 8 by DeepPolyR at [0, 4]=[-1.778802, 0.355760]
(DeepPolyR) Proved 2 7 8 by DeepPolyR at [2, 4]=[0.711521, 2.490323]
(Overall) Proved 2 7 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 9 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 7 9 by DeepPolyR at [0, 4]=[-2.134562, 0.000000]
(DeepPolyR) Proved 2 7 9 by DeepPolyR at [2, 4]=[0.711521, 2.134562]
(Overall) Proved 2 7 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 0 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 8 0 by DeepPolyR at [0, 4]=[-2.134562, 0.000000]
(DeepPolyR) Proved 2 8 0 by DeepPolyR at [2, 4]=[0.711521, 2.134562]
(Overall) Proved 2 8 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 1 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 8 1 by DeepPolyR at [0, 4]=[-2.490323, -2.490323]
(DeepPolyR) Proved 2 8 1 by DeepPolyR at [1, 4]=[0.000000, 1.778802]
(Overall) Proved 2 8 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 2 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 8 2 by DeepPolyR at [0, 4]=[-2.134562, -0.355760]
(DeepPolyR) Proved 2 8 2 by DeepPolyR at [4, 4]=[2.134562, 2.134562]
(Overall) Proved 2 8 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 3 by DeepPolyR at [0, 4]=[-1.423042, 2.134562]
(DeepPolyR) Proved 2 8 3 by DeepPolyR at [0, 4]=[-1.423042, 0.000000]
(DeepPolyR) Proved 2 8 3 by DeepPolyR at [2, 4]=[1.423042, 2.134562]
(Overall) Proved 2 8 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 4 by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 8 4 by DeepPolyR at [0, 4]=[-2.846083, -2.846083]
(DeepPolyR) Proved 2 8 4 by DeepPolyR at [1, 4]=[0.000000, 1.423042]
(Overall) Proved 2 8 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 5 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 8 5 by DeepPolyR at [0, 4]=[-2.134562, -0.355760]
(DeepPolyR) Proved 2 8 5 by DeepPolyR at [4, 4]=[2.134562, 2.134562]
(Overall) Proved 2 8 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 6 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 8 6 by DeepPolyR at [0, 4]=[-2.134562, 0.000000]
(DeepPolyR) Proved 2 8 6 by DeepPolyR at [2, 4]=[0.711521, 2.134562]
(Overall) Proved 2 8 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 7 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 8 7 by DeepPolyR at [0, 4]=[-1.778802, 0.355760]
(DeepPolyR) Proved 2 8 7 by DeepPolyR at [2, 4]=[0.711521, 2.490323]
(Overall) Proved 2 8 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 8 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 8 8 by DeepPolyR at [0, 4]=[-1.778802, 0.355760]
(DeepPolyR) Proved 2 8 8 by DeepPolyR at [2, 4]=[0.711521, 2.490323]
(Overall) Proved 2 8 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 9 by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 8 9 by DeepPolyR at [0, 4]=[-2.846083, -2.846083]
(DeepPolyR) Proved 2 8 9 by DeepPolyR at [1, 4]=[0.000000, 1.423042]
(Overall) Proved 2 8 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 0 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 9 0 by DeepPolyR at [0, 4]=[-2.490323, -1.067281]
(DeepPolyR) Proved 2 9 0 by DeepPolyR at [3, 4]=[-0.355760, 1.778802]
(Overall) Proved 2 9 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 1 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 9 1 by DeepPolyR at [0, 4]=[-1.778802, -0.355760]
(DeepPolyR) Proved 2 9 1 by DeepPolyR at [4, 4]=[2.490323, 2.490323]
(Overall) Proved 2 9 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 2 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 9 2 by DeepPolyR at [0, 4]=[-2.490323, -2.490323]
(DeepPolyR) Proved 2 9 2 by DeepPolyR at [1, 4]=[0.000000, 1.778802]
(Overall) Proved 2 9 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 3 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 9 3 by DeepPolyR at [0, 4]=[-2.490323, -1.067281]
(DeepPolyR) Proved 2 9 3 by DeepPolyR at [3, 4]=[-0.355760, 1.778802]
(Overall) Proved 2 9 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 4 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 9 4 by DeepPolyR at [0, 4]=[-2.134562, -0.355760]
(DeepPolyR) Proved 2 9 4 by DeepPolyR at [4, 4]=[2.134562, 2.134562]
(Overall) Proved 2 9 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 5 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 9 5 by DeepPolyR at [0, 4]=[-2.490323, -2.490323]
(DeepPolyR) Proved 2 9 5 by DeepPolyR at [1, 4]=[0.000000, 1.778802]
(Overall) Proved 2 9 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 6 by DeepPolyR at [0, 4]=[-1.778802, 2.490323]
(DeepPolyR) Proved 2 9 6 by DeepPolyR at [0, 4]=[-1.778802, -0.355760]
(DeepPolyR) Proved 2 9 6 by DeepPolyR at [4, 4]=[2.490323, 2.490323]
(Overall) Proved 2 9 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 7 by DeepPolyR at [0, 4]=[-2.134562, 2.134562]
(DeepPolyR) Proved 2 9 7 by DeepPolyR at [0, 4]=[-2.134562, -0.355760]
(DeepPolyR) Proved 2 9 7 by DeepPolyR at [4, 4]=[2.134562, 2.134562]
(Overall) Proved 2 9 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 8 by DeepPolyR at [0, 4]=[-2.846083, 1.423042]
(DeepPolyR) Proved 2 9 8 by DeepPolyR at [0, 4]=[-2.846083, -2.846083]
(DeepPolyR) Proved 2 9 8 by DeepPolyR at [1, 4]=[0.000000, 1.423042]
(Overall) Proved 2 9 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 9 by DeepPolyR at [0, 4]=[-2.490323, 1.778802]
(DeepPolyR) Proved 2 9 9 by DeepPolyR at [0, 4]=[-2.490323, -1.067281]
(DeepPolyR) Proved 2 9 9 by DeepPolyR at [3, 4]=[-0.355760, 1.778802]
(Overall) Proved 2 9 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 0 by DeepPolyR at [0, 4]=[-2.774479, 1.387240]
(DeepPolyR) Proved 3 0 0 by DeepPolyR at [0, 4]=[-2.774479, -2.774479]
(DeepPolyR) Proved 3 0 0 by DeepPolyR at [1, 4]=[0.000000, 1.387240]
(Overall) Proved 3 0 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 1 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 0 1 by DeepPolyR at [0, 4]=[-1.734049, -1.734049]
(DeepPolyR) Proved 3 0 1 by DeepPolyR at [1, 4]=[0.346810, 2.427669]
(Overall) Proved 3 0 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 2 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 0 2 by DeepPolyR at [0, 4]=[-2.080859, 0.000000]
(DeepPolyR) Proved 3 0 2 by DeepPolyR at [2, 4]=[0.693620, 2.080859]
(Overall) Proved 3 0 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 3 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 0 3 by DeepPolyR at [0, 4]=[-2.080859, -0.346810]
(DeepPolyR) Proved 3 0 3 by DeepPolyR at [4, 4]=[2.080859, 2.080859]
(Overall) Proved 3 0 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 4 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 0 4 by DeepPolyR at [0, 4]=[-2.080859, 0.000000]
(DeepPolyR) Proved 3 0 4 by DeepPolyR at [2, 4]=[0.693620, 2.080859]
(Overall) Proved 3 0 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 5 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 0 5 by DeepPolyR at [0, 4]=[-2.080859, -0.346810]
(DeepPolyR) Proved 3 0 5 by DeepPolyR at [4, 4]=[2.080859, 2.080859]
(Overall) Proved 3 0 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 6 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 0 6 by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 0 6 by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 0 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 7 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 0 7 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 0 7 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 0 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 8 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 0 8 by DeepPolyR at [0, 4]=[-2.080859, -0.346810]
(DeepPolyR) Proved 3 0 8 by DeepPolyR at [4, 4]=[2.080859, 2.080859]
(Overall) Proved 3 0 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 9 by DeepPolyR at [0, 4]=[-2.774479, 1.387240]
(DeepPolyR) Proved 3 0 9 by DeepPolyR at [0, 4]=[-2.774479, -2.774479]
(DeepPolyR) Proved 3 0 9 by DeepPolyR at [1, 4]=[0.000000, 1.387240]
(Overall) Proved 3 0 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 0 by DeepPolyR at [0, 4]=[-2.774479, 1.387240]
(DeepPolyR) Proved 3 1 0 by DeepPolyR at [0, 4]=[-2.774479, -2.774479]
(DeepPolyR) Proved 3 1 0 by DeepPolyR at [1, 4]=[0.000000, 1.387240]
(Overall) Proved 3 1 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 1 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 1 1 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 1 1 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 1 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 2 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 1 2 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 1 2 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 1 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 3 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 1 3 by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 1 3 by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 1 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 4 by DeepPolyR at [0, 4]=[-1.040430, 2.427669]
(DeepPolyR) Proved 3 1 4 by DeepPolyR at [0, 4]=[-1.040430, 0.346810]
(DeepPolyR) Proved 3 1 4 by DeepPolyR at [2, 4]=[1.387240, 2.427669]
(Overall) Proved 3 1 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 5 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 1 5 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 1 5 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 1 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 6 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 1 6 by DeepPolyR at [0, 4]=[-2.080859, -0.346810]
(DeepPolyR) Proved 3 1 6 by DeepPolyR at [4, 4]=[2.080859, 2.080859]
(Overall) Proved 3 1 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 7 by DeepPolyR at [0, 4]=[-2.774479, 0.346810]
(DeepPolyR) Proved 3 1 7 by DeepPolyR at [0, 4]=[-2.774479, -1.734049]
(DeepPolyR) Proved 3 1 7 by DeepPolyR at [3, 4]=[-1.040430, 0.346810]
(Overall) Proved 3 1 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 8 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 1 8 by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 1 8 by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 1 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 9 by DeepPolyR at [0, 4]=[-2.774479, 1.387240]
(DeepPolyR) Proved 3 1 9 by DeepPolyR at [0, 4]=[-2.774479, -2.774479]
(DeepPolyR) Proved 3 1 9 by DeepPolyR at [1, 4]=[0.000000, 1.387240]
(Overall) Proved 3 1 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 3 2 0 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Fail to prove 3 2 0 by DeepPolyR at [0, 4]=[-2.080859, 0.000000]
(DeepPolyR) Proved 3 2 0 by DeepPolyR at [2, 4]=[0.693620, 2.080859]
(Divide and Counter) Fail to prove 3 2 0 by DeepPolyR at [0, 1]=[-2.080859, 0.000000]
(Divide and Counter) Proved 3 2 0 by DeepPoly at [0]=-2.080859
(Divide and Counter) Proved 3 2 0 by DeepPoly at [1]=0.000000
(Divide and Counter) Proved 3 2 0 by DeepPolyR at [2, 4]=[0.693620, 2.080859]
(Overall) Proved 3 2 0 with all masks. Summary: 1 0 1 

(Divide and Counter) Proved 3 2 1 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 2 1 by DeepPolyR at [0, 4]=[-2.080859, -0.346810]
(DeepPolyR) Proved 3 2 1 by DeepPolyR at [4, 4]=[2.080859, 2.080859]
(Overall) Proved 3 2 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 2 2 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 2 2 by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 2 2 by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 2 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 3 2 3 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Fail to prove 3 2 3 by DeepPolyR at [0, 4]=[-2.080859, 0.000000]
(DeepPolyR) Proved 3 2 3 by DeepPolyR at [2, 4]=[0.693620, 2.080859]
(Divide and Counter) Fail to prove 3 2 3 by DeepPolyR at [0, 1]=[-2.080859, 0.000000]
(Divide and Counter) Proved 3 2 3 by DeepPoly at [0]=-2.080859
(Divide and Counter) Proved 3 2 3 by DeepPoly at [1]=0.000000
(Divide and Counter) Proved 3 2 3 by DeepPolyR at [2, 4]=[0.693620, 2.080859]
(Overall) Proved 3 2 3 with all masks. Summary: 1 0 1 

(Divide and Counter) Proved 3 2 4 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 2 4 by DeepPolyR at [0, 4]=[-2.080859, -0.346810]
(DeepPolyR) Proved 3 2 4 by DeepPolyR at [4, 4]=[2.080859, 2.080859]
(Overall) Proved 3 2 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 2 5 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 2 5 by DeepPolyR at [0, 4]=[-2.080859, 0.000000]
(DeepPolyR) Proved 3 2 5 by DeepPolyR at [2, 4]=[0.693620, 2.080859]
(Overall) Proved 3 2 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 2 6 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 2 6 by DeepPolyR at [0, 4]=[-2.080859, 0.000000]
(DeepPolyR) Proved 3 2 6 by DeepPolyR at [2, 4]=[0.693620, 2.080859]
(Overall) Proved 3 2 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 2 7 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 2 7 by DeepPolyR at [0, 4]=[-2.080859, 0.000000]
(DeepPolyR) Proved 3 2 7 by DeepPolyR at [2, 4]=[0.693620, 2.080859]
(Overall) Proved 3 2 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 2 8 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 2 8 by DeepPolyR at [0, 4]=[-2.080859, 0.000000]
(DeepPolyR) Proved 3 2 8 by DeepPolyR at [2, 4]=[0.693620, 2.080859]
(Overall) Proved 3 2 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 2 9 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 2 9 by DeepPolyR at [0, 4]=[-2.080859, 0.000000]
(DeepPolyR) Proved 3 2 9 by DeepPolyR at [2, 4]=[0.693620, 2.080859]
(Overall) Proved 3 2 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 0 by DeepPolyR at [0, 4]=[-0.693620, 2.427669]
(DeepPolyR) Proved 3 3 0 by DeepPolyR at [0, 4]=[-0.693620, 0.693620]
(DeepPolyR) Proved 3 3 0 by DeepPolyR at [2, 4]=[1.387240, 2.427669]
(Overall) Proved 3 3 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 1 by DeepPolyR at [0, 4]=[-0.346810, 2.427669]
(DeepPolyR) Proved 3 3 1 by DeepPolyR at [0, 4]=[-0.346810, -0.346810]
(DeepPolyR) Proved 3 3 1 by DeepPolyR at [1, 4]=[1.040430, 2.427669]
(Overall) Proved 3 3 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 2 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 3 2 by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 3 2 by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 3 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 3 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 3 3 by DeepPolyR at [0, 4]=[-1.734049, -1.734049]
(DeepPolyR) Proved 3 3 3 by DeepPolyR at [1, 4]=[0.346810, 2.427669]
(Overall) Proved 3 3 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 4 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 3 4 by DeepPolyR at [0, 4]=[-2.427669, -0.346810]
(DeepPolyR) Proved 3 3 4 by DeepPolyR at [4, 4]=[1.734049, 1.734049]
(Overall) Proved 3 3 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 5 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 3 5 by DeepPolyR at [0, 4]=[-2.080859, -0.346810]
(DeepPolyR) Proved 3 3 5 by DeepPolyR at [4, 4]=[2.080859, 2.080859]
(Overall) Proved 3 3 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 6 by DeepPolyR at [0, 4]=[-2.774479, 1.387240]
(DeepPolyR) Proved 3 3 6 by DeepPolyR at [0, 4]=[-2.774479, -2.774479]
(DeepPolyR) Proved 3 3 6 by DeepPolyR at [1, 4]=[0.000000, 1.387240]
(Overall) Proved 3 3 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 7 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 3 7 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 3 7 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 3 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 8 by DeepPolyR at [0, 4]=[-2.774479, 1.387240]
(DeepPolyR) Proved 3 3 8 by DeepPolyR at [0, 4]=[-2.774479, -2.774479]
(DeepPolyR) Proved 3 3 8 by DeepPolyR at [1, 4]=[0.000000, 1.387240]
(Overall) Proved 3 3 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 9 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 3 9 by DeepPolyR at [0, 4]=[-2.080859, 0.000000]
(DeepPolyR) Proved 3 3 9 by DeepPolyR at [2, 4]=[0.693620, 2.080859]
(Overall) Proved 3 3 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 0 by DeepPolyR at [0, 4]=[-2.774479, 1.387240]
(DeepPolyR) Proved 3 4 0 by DeepPolyR at [0, 4]=[-2.774479, -2.774479]
(DeepPolyR) Proved 3 4 0 by DeepPolyR at [1, 4]=[0.000000, 1.387240]
(Overall) Proved 3 4 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 1 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 4 1 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 4 1 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 4 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 2 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 4 2 by DeepPolyR at [0, 4]=[-1.734049, -1.734049]
(DeepPolyR) Proved 3 4 2 by DeepPolyR at [1, 4]=[0.346810, 2.427669]
(Overall) Proved 3 4 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 3 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 4 3 by DeepPolyR at [0, 4]=[-2.080859, -0.346810]
(DeepPolyR) Proved 3 4 3 by DeepPolyR at [4, 4]=[2.080859, 2.080859]
(Overall) Proved 3 4 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 4 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 4 4 by DeepPolyR at [0, 4]=[-2.080859, 0.000000]
(DeepPolyR) Proved 3 4 4 by DeepPolyR at [2, 4]=[0.693620, 2.080859]
(Overall) Proved 3 4 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 5 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 4 5 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 4 5 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 4 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 6 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 4 6 by DeepPolyR at [0, 4]=[-2.080859, -0.346810]
(DeepPolyR) Proved 3 4 6 by DeepPolyR at [4, 4]=[2.080859, 2.080859]
(Overall) Proved 3 4 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 7 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 4 7 by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 4 7 by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 4 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 8 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 4 8 by DeepPolyR at [0, 4]=[-2.427669, -0.346810]
(DeepPolyR) Proved 3 4 8 by DeepPolyR at [4, 4]=[1.734049, 1.734049]
(Overall) Proved 3 4 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 9 by DeepPolyR at [0, 4]=[-2.774479, 1.387240]
(DeepPolyR) Proved 3 4 9 by DeepPolyR at [0, 4]=[-2.774479, -2.774479]
(DeepPolyR) Proved 3 4 9 by DeepPolyR at [1, 4]=[0.000000, 1.387240]
(Overall) Proved 3 4 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 3 5 0 by DeepPolyR at [0, 4]=[-2.774479, 1.387240]
(DeepPolyR) Proved 3 5 0 by DeepPolyR at [0, 4]=[-2.774479, -2.774479]
(DeepPolyR) Proved 3 5 0 by DeepPolyR at [1, 4]=[0.000000, 1.387240]
(Divide and Counter) Proved 3 5 0 by DeepPoly at [0]=-2.774479
(Divide and Counter) Proved 3 5 0 by DeepPolyR at [1, 4]=[0.000000, 1.387240]
(Overall) Proved 3 5 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 1 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 5 1 by DeepPolyR at [0, 4]=[-2.427669, -0.346810]
(DeepPolyR) Proved 3 5 1 by DeepPolyR at [4, 4]=[1.734049, 1.734049]
(Overall) Proved 3 5 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 2 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 5 2 by DeepPolyR at [0, 4]=[-2.080859, 0.000000]
(DeepPolyR) Proved 3 5 2 by DeepPolyR at [2, 4]=[0.693620, 2.080859]
(Overall) Proved 3 5 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 3 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 5 3 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 5 3 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 5 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 4 by DeepPolyR at [0, 4]=[-2.774479, 1.387240]
(DeepPolyR) Proved 3 5 4 by DeepPolyR at [0, 4]=[-2.774479, -0.693620]
(DeepPolyR) Proved 3 5 4 by DeepPolyR at [4, 4]=[1.387240, 1.387240]
(Overall) Proved 3 5 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 5 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 5 5 by DeepPolyR at [0, 4]=[-2.080859, 0.000000]
(DeepPolyR) Proved 3 5 5 by DeepPolyR at [2, 4]=[0.693620, 2.080859]
(Overall) Proved 3 5 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 6 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 5 6 by DeepPolyR at [0, 4]=[-2.080859, 0.000000]
(DeepPolyR) Proved 3 5 6 by DeepPolyR at [2, 4]=[0.693620, 2.080859]
(Overall) Proved 3 5 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 7 by DeepPolyR at [0, 4]=[-1.387240, 2.080859]
(DeepPolyR) Proved 3 5 7 by DeepPolyR at [0, 4]=[-1.387240, 0.000000]
(DeepPolyR) Proved 3 5 7 by DeepPolyR at [2, 4]=[1.387240, 2.080859]
(Overall) Proved 3 5 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 8 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 5 8 by DeepPolyR at [0, 4]=[-1.734049, -1.734049]
(DeepPolyR) Proved 3 5 8 by DeepPolyR at [1, 4]=[0.346810, 2.427669]
(Overall) Proved 3 5 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 9 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 5 9 by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 5 9 by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 5 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 0 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 6 0 by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 6 0 by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 6 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 1 by DeepPolyR at [0, 4]=[-2.774479, 1.387240]
(DeepPolyR) Proved 3 6 1 by DeepPolyR at [0, 4]=[-2.774479, -2.774479]
(DeepPolyR) Proved 3 6 1 by DeepPolyR at [1, 4]=[0.000000, 1.387240]
(Overall) Proved 3 6 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 2 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 6 2 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 6 2 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 6 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 3 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 6 3 by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 6 3 by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 6 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 4 by DeepPolyR at [0, 4]=[-1.387240, 2.080859]
(DeepPolyR) Proved 3 6 4 by DeepPolyR at [0, 4]=[-1.387240, 0.000000]
(DeepPolyR) Proved 3 6 4 by DeepPolyR at [2, 4]=[1.387240, 2.080859]
(Overall) Proved 3 6 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 5 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 6 5 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 6 5 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 6 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 6 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 6 6 by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 6 6 by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 6 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 7 by DeepPolyR at [0, 4]=[-2.774479, 1.387240]
(DeepPolyR) Proved 3 6 7 by DeepPolyR at [0, 4]=[-2.774479, -2.774479]
(DeepPolyR) Proved 3 6 7 by DeepPolyR at [1, 4]=[0.000000, 1.387240]
(Overall) Proved 3 6 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 8 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 6 8 by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 6 8 by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 6 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 9 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 6 9 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 6 9 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 6 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 3 7 0 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 7 0 by DeepPolyR at [0, 4]=[-2.080859, -0.346810]
(DeepPolyR) Fail to prove 3 7 0 by DeepPolyR at [4, 4]=[2.080859, 2.080859]
(Divide and Counter) Proved 3 7 0 by DeepPolyR at [0, 3]=[-2.080859, -0.346810]
(Divide and Counter) Fail to prove 3 7 0 by DeepPoly at [4]=2.080859
(Overall) Fail to prove 3 7 0 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 3 7 1 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 7 1 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 7 1 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 7 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 7 2 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 7 2 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 7 2 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 7 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 3 7 3 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 7 3 by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Fail to prove 3 7 3 by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Divide and Counter) Proved 3 7 3 by DeepPolyR at [0, 3]=[-1.734049, -0.346810]
(Divide and Counter) Fail to prove 3 7 3 by DeepPoly at [4]=2.427669
(Overall) Fail to prove 3 7 3 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 3 7 4 by DeepPolyR at [0, 4]=[-0.693620, 2.427669]
(DeepPolyR) Proved 3 7 4 by DeepPolyR at [0, 4]=[-0.693620, 0.693620]
(DeepPolyR) Proved 3 7 4 by DeepPolyR at [2, 4]=[1.387240, 2.427669]
(Overall) Proved 3 7 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 7 5 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 7 5 by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 7 5 by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 7 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 7 6 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 7 6 by DeepPolyR at [0, 4]=[-2.080859, -0.346810]
(DeepPolyR) Proved 3 7 6 by DeepPolyR at [4, 4]=[2.080859, 2.080859]
(Overall) Proved 3 7 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 7 7 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 7 7 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 7 7 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 7 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 7 8 by DeepPolyR at [0, 4]=[-2.774479, 1.387240]
(DeepPolyR) Proved 3 7 8 by DeepPolyR at [0, 4]=[-2.774479, -2.774479]
(DeepPolyR) Proved 3 7 8 by DeepPolyR at [1, 4]=[0.000000, 1.387240]
(Overall) Proved 3 7 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 7 9 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 7 9 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 7 9 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 7 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 8 0 by DeepPolyR at [0, 4]=[-2.774479, 1.387240]
(DeepPolyR) Proved 3 8 0 by DeepPolyR at [0, 4]=[-2.774479, -2.774479]
(DeepPolyR) Proved 3 8 0 by DeepPolyR at [1, 4]=[0.000000, 1.387240]
(Overall) Proved 3 8 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 8 1 by DeepPolyR at [0, 4]=[-1.387240, 2.080859]
(DeepPolyR) Proved 3 8 1 by DeepPolyR at [0, 4]=[-1.387240, 0.000000]
(DeepPolyR) Proved 3 8 1 by DeepPolyR at [2, 4]=[1.387240, 2.080859]
(Overall) Proved 3 8 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 8 2 by DeepPolyR at [0, 4]=[-2.774479, 1.387240]
(DeepPolyR) Proved 3 8 2 by DeepPolyR at [0, 4]=[-2.774479, -2.774479]
(DeepPolyR) Proved 3 8 2 by DeepPolyR at [1, 4]=[0.000000, 1.387240]
(Overall) Proved 3 8 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 8 3 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 8 3 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 8 3 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 8 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 8 4 by DeepPolyR at [0, 4]=[-1.387240, 2.080859]
(DeepPolyR) Proved 3 8 4 by DeepPolyR at [0, 4]=[-1.387240, 0.000000]
(DeepPolyR) Proved 3 8 4 by DeepPolyR at [2, 4]=[1.387240, 2.080859]
(Overall) Proved 3 8 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 8 5 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 8 5 by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 8 5 by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 8 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 8 6 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 8 6 by DeepPolyR at [0, 4]=[-2.080859, -0.346810]
(DeepPolyR) Proved 3 8 6 by DeepPolyR at [4, 4]=[2.080859, 2.080859]
(Overall) Proved 3 8 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 8 7 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 8 7 by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 8 7 by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 8 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 8 8 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 8 8 by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 8 8 by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 8 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 8 9 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 8 9 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 8 9 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 8 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 0 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 9 0 by DeepPolyR at [0, 4]=[-2.427669, -0.346810]
(DeepPolyR) Proved 3 9 0 by DeepPolyR at [4, 4]=[1.734049, 1.734049]
(Overall) Proved 3 9 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 1 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 9 1 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 9 1 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 9 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 2 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 9 2 by DeepPolyR at [0, 4]=[-1.734049, -1.734049]
(DeepPolyR) Proved 3 9 2 by DeepPolyR at [1, 4]=[0.346810, 2.427669]
(Overall) Proved 3 9 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 3 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 9 3 by DeepPolyR at [0, 4]=[-2.427669, -0.346810]
(DeepPolyR) Proved 3 9 3 by DeepPolyR at [4, 4]=[1.734049, 1.734049]
(Overall) Proved 3 9 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 4 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 9 4 by DeepPolyR at [0, 4]=[-2.080859, 0.000000]
(DeepPolyR) Proved 3 9 4 by DeepPolyR at [2, 4]=[0.693620, 2.080859]
(Overall) Proved 3 9 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 5 by DeepPolyR at [0, 4]=[-2.427669, 1.734049]
(DeepPolyR) Proved 3 9 5 by DeepPolyR at [0, 4]=[-2.427669, -2.427669]
(DeepPolyR) Proved 3 9 5 by DeepPolyR at [1, 4]=[0.000000, 1.734049]
(Overall) Proved 3 9 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 6 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 9 6 by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 9 6 by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 9 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 7 by DeepPolyR at [0, 4]=[-1.734049, 2.427669]
(DeepPolyR) Proved 3 9 7 by DeepPolyR at [0, 4]=[-1.734049, -0.346810]
(DeepPolyR) Proved 3 9 7 by DeepPolyR at [4, 4]=[2.427669, 2.427669]
(Overall) Proved 3 9 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 8 by DeepPolyR at [0, 4]=[-2.774479, 1.387240]
(DeepPolyR) Proved 3 9 8 by DeepPolyR at [0, 4]=[-2.774479, -2.774479]
(DeepPolyR) Proved 3 9 8 by DeepPolyR at [1, 4]=[0.000000, 1.387240]
(Overall) Proved 3 9 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 9 by DeepPolyR at [0, 4]=[-2.080859, 2.080859]
(DeepPolyR) Proved 3 9 9 by DeepPolyR at [0, 4]=[-2.080859, -0.346810]
(DeepPolyR) Proved 3 9 9 by DeepPolyR at [4, 4]=[2.080859, 2.080859]
(Overall) Proved 3 9 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 0 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 0 0 by DeepPolyR at [0, 4]=[-3.609603, 0.000000]
(DeepPolyR) Proved 4 0 0 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Overall) Proved 4 0 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 1 by DeepPolyR at [0, 4]=[-1.804802, 4.211204]
(DeepPolyR) Proved 4 0 1 by DeepPolyR at [0, 4]=[-1.804802, 0.601601]
(DeepPolyR) Proved 4 0 1 by DeepPolyR at [2, 4]=[2.406402, 4.211204]
(Overall) Proved 4 0 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 2 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 0 2 by DeepPolyR at [0, 4]=[-4.211204, -1.804802]
(DeepPolyR) Proved 4 0 2 by DeepPolyR at [3, 4]=[-0.601601, 3.008003]
(Overall) Proved 4 0 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 3 by DeepPolyR at [0, 4]=[-4.211204, 1.804802]
(DeepPolyR) Proved 4 0 3 by DeepPolyR at [0, 4]=[-4.211204, -3.008003]
(DeepPolyR) Proved 4 0 3 by DeepPolyR at [3, 4]=[-0.601601, 1.804802]
(Overall) Proved 4 0 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 4 by DeepPolyR at [0, 4]=[-2.406402, 3.609603]
(DeepPolyR) Proved 4 0 4 by DeepPolyR at [0, 4]=[-2.406402, 0.000000]
(DeepPolyR) Proved 4 0 4 by DeepPolyR at [2, 4]=[2.406402, 3.609603]
(Overall) Proved 4 0 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 5 by DeepPolyR at [0, 4]=[-4.211204, 1.804802]
(DeepPolyR) Proved 4 0 5 by DeepPolyR at [0, 4]=[-4.211204, -3.008003]
(DeepPolyR) Proved 4 0 5 by DeepPolyR at [3, 4]=[-0.601601, 1.804802]
(Overall) Proved 4 0 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 6 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 0 6 by DeepPolyR at [0, 4]=[-4.211204, -4.211204]
(DeepPolyR) Proved 4 0 6 by DeepPolyR at [1, 4]=[0.000000, 3.008003]
(Overall) Proved 4 0 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 7 by DeepPolyR at [0, 4]=[-1.804802, 4.211204]
(DeepPolyR) Proved 4 0 7 by DeepPolyR at [0, 4]=[-1.804802, 0.601601]
(DeepPolyR) Proved 4 0 7 by DeepPolyR at [2, 4]=[2.406402, 4.211204]
(Overall) Proved 4 0 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 8 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 0 8 by DeepPolyR at [0, 4]=[-4.211204, -4.211204]
(DeepPolyR) Proved 4 0 8 by DeepPolyR at [1, 4]=[0.000000, 3.008003]
(Overall) Proved 4 0 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 9 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 0 9 by DeepPolyR at [0, 4]=[-3.609603, 0.000000]
(DeepPolyR) Proved 4 0 9 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Overall) Proved 4 0 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 1 0 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 1 0 by DeepPolyR at [0, 4]=[-4.211204, -1.804802]
(DeepPolyR) Proved 4 1 0 by DeepPolyR at [3, 4]=[-0.601601, 3.008003]
(Overall) Proved 4 1 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 1 1 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 1 1 by DeepPolyR at [0, 4]=[-3.609603, -0.601601]
(DeepPolyR) Proved 4 1 1 by DeepPolyR at [4, 4]=[3.609603, 3.609603]
(Overall) Proved 4 1 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 1 2 by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Fail to prove 4 1 2 by DeepPolyR at [0, 4]=[-3.008003, 0.601601]
(DeepPolyR) Proved 4 1 2 by DeepPolyR at [2, 4]=[1.203201, 4.211204]
(Divide and Counter) Fail to prove 4 1 2 by DeepPolyR at [0, 1]=[-3.008003, 0.601601]
(Divide and Counter) Fail to prove 4 1 2 by DeepPoly at [0]=-3.008003
(Overall) Fail to prove 4 1 2 with all masks. Summary: 0 0 1 

(Divide and Counter) Fail to prove 4 1 3 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Fail to prove 4 1 3 by DeepPolyR at [0, 4]=[-3.609603, 0.000000]
(DeepPolyR) Proved 4 1 3 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Divide and Counter) Fail to prove 4 1 3 by DeepPolyR at [0, 1]=[-3.609603, 0.000000]
(Divide and Counter) Fail to prove 4 1 3 by DeepPoly at [0]=-3.609603
(Overall) Fail to prove 4 1 3 with all masks. Summary: 0 0 1 

(Divide and Counter) Proved 4 1 4 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 1 4 by DeepPolyR at [0, 4]=[-4.211204, -1.804802]
(DeepPolyR) Proved 4 1 4 by DeepPolyR at [3, 4]=[-0.601601, 3.008003]
(Overall) Proved 4 1 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 1 5 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Fail to prove 4 1 5 by DeepPolyR at [0, 4]=[-3.609603, 0.000000]
(DeepPolyR) Proved 4 1 5 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Divide and Counter) Fail to prove 4 1 5 by DeepPolyR at [0, 1]=[-3.609603, 0.000000]
(Divide and Counter) Fail to prove 4 1 5 by DeepPoly at [0]=-3.609603
(Overall) Fail to prove 4 1 5 with all masks. Summary: 0 0 1 

(Divide and Counter) Proved 4 1 6 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 1 6 by DeepPolyR at [0, 4]=[-4.211204, -1.804802]
(DeepPolyR) Proved 4 1 6 by DeepPolyR at [3, 4]=[-0.601601, 3.008003]
(Overall) Proved 4 1 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 1 7 by DeepPolyR at [0, 4]=[-4.812805, 1.203201]
(DeepPolyR) Proved 4 1 7 by DeepPolyR at [0, 4]=[-4.812805, -3.008003]
(DeepPolyR) Proved 4 1 7 by DeepPolyR at [3, 4]=[-1.203201, 1.203201]
(Overall) Proved 4 1 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 1 8 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 1 8 by DeepPolyR at [0, 4]=[-4.211204, -1.804802]
(DeepPolyR) Proved 4 1 8 by DeepPolyR at [3, 4]=[-0.601601, 3.008003]
(Overall) Proved 4 1 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 1 9 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 1 9 by DeepPolyR at [0, 4]=[-4.812805, -1.203201]
(DeepPolyR) Proved 4 1 9 by DeepPolyR at [4, 4]=[2.406402, 2.406402]
(Overall) Proved 4 1 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 0 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 2 0 by DeepPolyR at [0, 4]=[-4.812805, -4.812805]
(DeepPolyR) Proved 4 2 0 by DeepPolyR at [1, 4]=[0.000000, 2.406402]
(Overall) Proved 4 2 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 1 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 2 1 by DeepPolyR at [0, 4]=[-4.812805, -4.812805]
(DeepPolyR) Proved 4 2 1 by DeepPolyR at [1, 4]=[0.000000, 2.406402]
(Overall) Proved 4 2 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 2 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 2 2 by DeepPolyR at [0, 4]=[-3.609603, -0.601601]
(DeepPolyR) Proved 4 2 2 by DeepPolyR at [4, 4]=[3.609603, 3.609603]
(Overall) Proved 4 2 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 2 3 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 2 3 by DeepPolyR at [0, 4]=[-3.609603, -0.601601]
(DeepPolyR) Fail to prove 4 2 3 by DeepPolyR at [4, 4]=[3.609603, 3.609603]
(Divide and Counter) Proved 4 2 3 by DeepPolyR at [0, 3]=[-3.609603, -0.601601]
(Divide and Counter) Fail to prove 4 2 3 by DeepPoly at [4]=3.609603
(Overall) Fail to prove 4 2 3 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 4 2 4 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 2 4 by DeepPolyR at [0, 4]=[-3.609603, 0.000000]
(DeepPolyR) Proved 4 2 4 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Overall) Proved 4 2 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 5 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 2 5 by DeepPolyR at [0, 4]=[-4.812805, -4.812805]
(DeepPolyR) Proved 4 2 5 by DeepPolyR at [1, 4]=[0.000000, 2.406402]
(Overall) Proved 4 2 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 6 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 2 6 by DeepPolyR at [0, 4]=[-4.211204, -4.211204]
(DeepPolyR) Proved 4 2 6 by DeepPolyR at [1, 4]=[0.000000, 3.008003]
(Overall) Proved 4 2 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 7 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 2 7 by DeepPolyR at [0, 4]=[-4.211204, -4.211204]
(DeepPolyR) Proved 4 2 7 by DeepPolyR at [1, 4]=[0.000000, 3.008003]
(Overall) Proved 4 2 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 8 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 2 8 by DeepPolyR at [0, 4]=[-4.211204, -4.211204]
(DeepPolyR) Proved 4 2 8 by DeepPolyR at [1, 4]=[0.000000, 3.008003]
(Overall) Proved 4 2 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 9 by DeepPolyR at [0, 4]=[-1.804802, 4.211204]
(DeepPolyR) Proved 4 2 9 by DeepPolyR at [0, 4]=[-1.804802, 0.601601]
(DeepPolyR) Proved 4 2 9 by DeepPolyR at [2, 4]=[2.406402, 4.211204]
(Overall) Proved 4 2 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 0 by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 3 0 by DeepPolyR at [0, 4]=[-3.008003, 0.601601]
(DeepPolyR) Proved 4 3 0 by DeepPolyR at [2, 4]=[1.203201, 4.211204]
(Overall) Proved 4 3 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 1 by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 3 1 by DeepPolyR at [0, 4]=[-3.008003, 0.601601]
(DeepPolyR) Proved 4 3 1 by DeepPolyR at [2, 4]=[1.203201, 4.211204]
(Overall) Proved 4 3 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 2 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 3 2 by DeepPolyR at [0, 4]=[-3.609603, -0.601601]
(DeepPolyR) Proved 4 3 2 by DeepPolyR at [4, 4]=[3.609603, 3.609603]
(Overall) Proved 4 3 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 3 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 3 3 by DeepPolyR at [0, 4]=[-4.211204, -4.211204]
(DeepPolyR) Proved 4 3 3 by DeepPolyR at [1, 4]=[0.000000, 3.008003]
(Overall) Proved 4 3 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 4 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 3 4 by DeepPolyR at [0, 4]=[-3.609603, 0.000000]
(DeepPolyR) Proved 4 3 4 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Overall) Proved 4 3 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 5 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 3 5 by DeepPolyR at [0, 4]=[-4.812805, -1.203201]
(DeepPolyR) Proved 4 3 5 by DeepPolyR at [4, 4]=[2.406402, 2.406402]
(Overall) Proved 4 3 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 6 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 3 6 by DeepPolyR at [0, 4]=[-4.812805, -4.812805]
(DeepPolyR) Proved 4 3 6 by DeepPolyR at [1, 4]=[0.000000, 2.406402]
(Overall) Proved 4 3 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 7 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 3 7 by DeepPolyR at [0, 4]=[-3.609603, 0.000000]
(DeepPolyR) Proved 4 3 7 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Overall) Proved 4 3 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 8 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 3 8 by DeepPolyR at [0, 4]=[-4.211204, -4.211204]
(DeepPolyR) Proved 4 3 8 by DeepPolyR at [1, 4]=[0.000000, 3.008003]
(Overall) Proved 4 3 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 9 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 3 9 by DeepPolyR at [0, 4]=[-3.609603, 0.000000]
(DeepPolyR) Proved 4 3 9 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Overall) Proved 4 3 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 4 0 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 4 0 by DeepPolyR at [0, 4]=[-3.609603, -0.601601]
(DeepPolyR) Proved 4 4 0 by DeepPolyR at [4, 4]=[3.609603, 3.609603]
(Overall) Proved 4 4 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 4 1 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 4 1 by DeepPolyR at [0, 4]=[-4.211204, -1.804802]
(DeepPolyR) Proved 4 4 1 by DeepPolyR at [3, 4]=[-0.601601, 3.008003]
(Overall) Proved 4 4 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 4 2 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 4 2 by DeepPolyR at [0, 4]=[-3.609603, 0.000000]
(DeepPolyR) Proved 4 4 2 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Overall) Proved 4 4 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 4 3 by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 4 3 by DeepPolyR at [0, 4]=[-3.008003, -0.601601]
(DeepPolyR) Fail to prove 4 4 3 by DeepPolyR at [4, 4]=[4.211204, 4.211204]
(Divide and Counter) Proved 4 4 3 by DeepPolyR at [0, 3]=[-3.008003, -0.601601]
(Divide and Counter) Fail to prove 4 4 3 by DeepPoly at [4]=4.211204
(Overall) Fail to prove 4 4 3 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 4 4 4 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 4 4 by DeepPolyR at [0, 4]=[-3.609603, -0.601601]
(DeepPolyR) Proved 4 4 4 by DeepPolyR at [4, 4]=[3.609603, 3.609603]
(Overall) Proved 4 4 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 4 5 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 4 5 by DeepPolyR at [0, 4]=[-3.609603, 0.000000]
(DeepPolyR) Proved 4 4 5 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Overall) Proved 4 4 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 4 6 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 4 6 by DeepPolyR at [0, 4]=[-4.211204, -1.804802]
(DeepPolyR) Proved 4 4 6 by DeepPolyR at [3, 4]=[-0.601601, 3.008003]
(Overall) Proved 4 4 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 4 7 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 4 7 by DeepPolyR at [0, 4]=[-4.812805, -1.203201]
(DeepPolyR) Proved 4 4 7 by DeepPolyR at [4, 4]=[2.406402, 2.406402]
(Overall) Proved 4 4 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 4 8 by DeepPolyR at [0, 4]=[-4.812805, 0.601601]
(DeepPolyR) Proved 4 4 8 by DeepPolyR at [0, 4]=[-4.812805, -3.008003]
(DeepPolyR) Proved 4 4 8 by DeepPolyR at [3, 4]=[-1.804802, 0.601601]
(Overall) Proved 4 4 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 4 9 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 4 9 by DeepPolyR at [0, 4]=[-4.812805, -4.812805]
(DeepPolyR) Proved 4 4 9 by DeepPolyR at [1, 4]=[0.000000, 2.406402]
(Overall) Proved 4 4 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 0 by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 5 0 by DeepPolyR at [0, 4]=[-3.008003, -0.601601]
(DeepPolyR) Proved 4 5 0 by DeepPolyR at [4, 4]=[4.211204, 4.211204]
(Overall) Proved 4 5 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 1 by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 5 1 by DeepPolyR at [0, 4]=[-3.008003, -0.601601]
(DeepPolyR) Proved 4 5 1 by DeepPolyR at [4, 4]=[4.211204, 4.211204]
(Overall) Proved 4 5 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 2 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 5 2 by DeepPolyR at [0, 4]=[-4.812805, -4.812805]
(DeepPolyR) Proved 4 5 2 by DeepPolyR at [1, 4]=[0.000000, 2.406402]
(Overall) Proved 4 5 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 5 3 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 5 3 by DeepPolyR at [0, 4]=[-4.812805, -4.812805]
(DeepPolyR) Fail to prove 4 5 3 by DeepPolyR at [1, 4]=[0.000000, 2.406402]
(Divide and Counter) Proved 4 5 3 by DeepPoly at [0]=-4.812805
(Divide and Counter) Fail to prove 4 5 3 by DeepPolyR at [1, 4]=[0.000000, 2.406402]
(Divide and Counter) Proved 4 5 3 by DeepPolyR at [1, 3]=[0.000000, 1.203201]
(Divide and Counter) Fail to prove 4 5 3 by DeepPoly at [4]=2.406402
(Overall) Fail to prove 4 5 3 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 4 5 4 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 5 4 by DeepPolyR at [0, 4]=[-4.812805, -4.812805]
(DeepPolyR) Proved 4 5 4 by DeepPolyR at [1, 4]=[0.000000, 2.406402]
(Overall) Proved 4 5 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 5 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 5 5 by DeepPolyR at [0, 4]=[-4.812805, -4.812805]
(DeepPolyR) Proved 4 5 5 by DeepPolyR at [1, 4]=[0.000000, 2.406402]
(Overall) Proved 4 5 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 6 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 5 6 by DeepPolyR at [0, 4]=[-4.812805, -4.812805]
(DeepPolyR) Proved 4 5 6 by DeepPolyR at [1, 4]=[0.000000, 2.406402]
(Overall) Proved 4 5 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 7 by DeepPolyR at [0, 4]=[-2.406402, 3.609603]
(DeepPolyR) Proved 4 5 7 by DeepPolyR at [0, 4]=[-2.406402, 0.000000]
(DeepPolyR) Proved 4 5 7 by DeepPolyR at [2, 4]=[2.406402, 3.609603]
(Overall) Proved 4 5 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 8 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 5 8 by DeepPolyR at [0, 4]=[-3.609603, 0.000000]
(DeepPolyR) Proved 4 5 8 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Overall) Proved 4 5 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 9 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 5 9 by DeepPolyR at [0, 4]=[-4.812805, -4.812805]
(DeepPolyR) Proved 4 5 9 by DeepPolyR at [1, 4]=[0.000000, 2.406402]
(Overall) Proved 4 5 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 0 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 6 0 by DeepPolyR at [0, 4]=[-3.609603, 0.000000]
(DeepPolyR) Proved 4 6 0 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Overall) Proved 4 6 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 1 by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 6 1 by DeepPolyR at [0, 4]=[-3.008003, -0.601601]
(DeepPolyR) Proved 4 6 1 by DeepPolyR at [4, 4]=[4.211204, 4.211204]
(Overall) Proved 4 6 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 2 by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 6 2 by DeepPolyR at [0, 4]=[-3.008003, -0.601601]
(DeepPolyR) Proved 4 6 2 by DeepPolyR at [4, 4]=[4.211204, 4.211204]
(Overall) Proved 4 6 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 3 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 6 3 by DeepPolyR at [0, 4]=[-4.812805, -4.812805]
(DeepPolyR) Proved 4 6 3 by DeepPolyR at [1, 4]=[0.000000, 2.406402]
(Overall) Proved 4 6 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 4 by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 6 4 by DeepPolyR at [0, 4]=[-3.008003, 0.601601]
(DeepPolyR) Proved 4 6 4 by DeepPolyR at [2, 4]=[1.203201, 4.211204]
(Overall) Proved 4 6 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 5 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 6 5 by DeepPolyR at [0, 4]=[-4.812805, -4.812805]
(DeepPolyR) Proved 4 6 5 by DeepPolyR at [1, 4]=[0.000000, 2.406402]
(Overall) Proved 4 6 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 6 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 6 6 by DeepPolyR at [0, 4]=[-4.211204, -4.211204]
(DeepPolyR) Proved 4 6 6 by DeepPolyR at [1, 4]=[0.000000, 3.008003]
(Overall) Proved 4 6 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 7 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 6 7 by DeepPolyR at [0, 4]=[-4.211204, -4.211204]
(DeepPolyR) Proved 4 6 7 by DeepPolyR at [1, 4]=[0.000000, 3.008003]
(Overall) Proved 4 6 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 8 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 6 8 by DeepPolyR at [0, 4]=[-4.812805, -4.812805]
(DeepPolyR) Proved 4 6 8 by DeepPolyR at [1, 4]=[0.000000, 2.406402]
(Overall) Proved 4 6 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 9 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 6 9 by DeepPolyR at [0, 4]=[-4.211204, -4.211204]
(DeepPolyR) Proved 4 6 9 by DeepPolyR at [1, 4]=[0.000000, 3.008003]
(Overall) Proved 4 6 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 7 0 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 7 0 by DeepPolyR at [0, 4]=[-4.812805, -1.203201]
(DeepPolyR) Proved 4 7 0 by DeepPolyR at [4, 4]=[2.406402, 2.406402]
(Overall) Proved 4 7 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 7 1 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 7 1 by DeepPolyR at [0, 4]=[-3.609603, -0.601601]
(DeepPolyR) Proved 4 7 1 by DeepPolyR at [4, 4]=[3.609603, 3.609603]
(Overall) Proved 4 7 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 7 2 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 7 2 by DeepPolyR at [0, 4]=[-3.609603, 0.000000]
(DeepPolyR) Fail to prove 4 7 2 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Divide and Counter) Proved 4 7 2 by DeepPolyR at [0, 1]=[-3.609603, 0.000000]
(Divide and Counter) Fail to prove 4 7 2 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Divide and Counter) Proved 4 7 2 by DeepPolyR at [2, 3]=[1.203201, 1.804802]
(Divide and Counter) Fail to prove 4 7 2 by DeepPoly at [4]=3.609603
(Overall) Fail to prove 4 7 2 with all masks. Summary: 0 1 0 

(Divide and Counter) Fail to prove 4 7 3 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 7 3 by DeepPolyR at [0, 4]=[-3.609603, 0.000000]
(DeepPolyR) Fail to prove 4 7 3 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Divide and Counter) Proved 4 7 3 by DeepPolyR at [0, 1]=[-3.609603, 0.000000]
(Divide and Counter) Fail to prove 4 7 3 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Divide and Counter) Fail to prove 4 7 3 by DeepPolyR at [2, 3]=[1.203201, 1.804802]
(Divide and Counter) Proved 4 7 3 by DeepPoly at [2]=1.203201
(Divide and Counter) Fail to prove 4 7 3 by DeepPoly at [3]=1.804802
(Overall) Fail to prove 4 7 3 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 4 7 4 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 7 4 by DeepPolyR at [0, 4]=[-4.812805, -1.203201]
(DeepPolyR) Proved 4 7 4 by DeepPolyR at [4, 4]=[2.406402, 2.406402]
(Overall) Proved 4 7 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 7 5 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 7 5 by DeepPolyR at [0, 4]=[-4.211204, -4.211204]
(DeepPolyR) Fail to prove 4 7 5 by DeepPolyR at [1, 4]=[0.000000, 3.008003]
(Divide and Counter) Proved 4 7 5 by DeepPoly at [0]=-4.211204
(Divide and Counter) Fail to prove 4 7 5 by DeepPolyR at [1, 4]=[0.000000, 3.008003]
(Divide and Counter) Proved 4 7 5 by DeepPolyR at [1, 2]=[0.000000, 0.601601]
(Divide and Counter) Fail to prove 4 7 5 by DeepPolyR at [3, 4]=[1.804802, 3.008003]
(Divide and Counter) Proved 4 7 5 by DeepPoly at [3]=1.804802
(Divide and Counter) Fail to prove 4 7 5 by DeepPoly at [4]=3.008003
(Overall) Fail to prove 4 7 5 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 4 7 6 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 7 6 by DeepPolyR at [0, 4]=[-3.609603, -0.601601]
(DeepPolyR) Proved 4 7 6 by DeepPolyR at [4, 4]=[3.609603, 3.609603]
(Overall) Proved 4 7 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 7 7 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 7 7 by DeepPolyR at [0, 4]=[-4.211204, -1.804802]
(DeepPolyR) Proved 4 7 7 by DeepPolyR at [3, 4]=[-0.601601, 3.008003]
(Overall) Proved 4 7 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 7 8 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 7 8 by DeepPolyR at [0, 4]=[-3.609603, -0.601601]
(DeepPolyR) Proved 4 7 8 by DeepPolyR at [4, 4]=[3.609603, 3.609603]
(Overall) Proved 4 7 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 7 9 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 7 9 by DeepPolyR at [0, 4]=[-4.211204, -1.804802]
(DeepPolyR) Proved 4 7 9 by DeepPolyR at [3, 4]=[-0.601601, 3.008003]
(Overall) Proved 4 7 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 0 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 8 0 by DeepPolyR at [0, 4]=[-4.211204, -4.211204]
(DeepPolyR) Proved 4 8 0 by DeepPolyR at [1, 4]=[0.000000, 3.008003]
(Overall) Proved 4 8 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 1 by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 8 1 by DeepPolyR at [0, 4]=[-3.008003, -0.601601]
(DeepPolyR) Proved 4 8 1 by DeepPolyR at [4, 4]=[4.211204, 4.211204]
(Overall) Proved 4 8 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 8 2 by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 8 2 by DeepPolyR at [0, 4]=[-3.008003, -0.601601]
(DeepPolyR) Fail to prove 4 8 2 by DeepPolyR at [4, 4]=[4.211204, 4.211204]
(Divide and Counter) Proved 4 8 2 by DeepPolyR at [0, 3]=[-3.008003, -0.601601]
(Divide and Counter) Fail to prove 4 8 2 by DeepPoly at [4]=4.211204
(Overall) Fail to prove 4 8 2 with all masks. Summary: 0 1 0 

(Divide and Counter) Fail to prove 4 8 3 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 8 3 by DeepPolyR at [0, 4]=[-3.609603, 0.000000]
(DeepPolyR) Fail to prove 4 8 3 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Divide and Counter) Proved 4 8 3 by DeepPolyR at [0, 1]=[-3.609603, 0.000000]
(Divide and Counter) Fail to prove 4 8 3 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Divide and Counter) Proved 4 8 3 by DeepPolyR at [2, 3]=[1.203201, 1.804802]
(Divide and Counter) Fail to prove 4 8 3 by DeepPoly at [4]=3.609603
(Overall) Fail to prove 4 8 3 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 4 8 4 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 8 4 by DeepPolyR at [0, 4]=[-4.211204, -4.211204]
(DeepPolyR) Proved 4 8 4 by DeepPolyR at [1, 4]=[0.000000, 3.008003]
(Overall) Proved 4 8 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 5 by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 8 5 by DeepPolyR at [0, 4]=[-3.008003, -0.601601]
(DeepPolyR) Proved 4 8 5 by DeepPolyR at [4, 4]=[4.211204, 4.211204]
(Overall) Proved 4 8 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 6 by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 8 6 by DeepPolyR at [0, 4]=[-3.008003, -0.601601]
(DeepPolyR) Proved 4 8 6 by DeepPolyR at [4, 4]=[4.211204, 4.211204]
(Overall) Proved 4 8 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 7 by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 8 7 by DeepPolyR at [0, 4]=[-3.008003, -0.601601]
(DeepPolyR) Proved 4 8 7 by DeepPolyR at [4, 4]=[4.211204, 4.211204]
(Overall) Proved 4 8 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 8 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 8 8 by DeepPolyR at [0, 4]=[-4.812805, -4.812805]
(DeepPolyR) Proved 4 8 8 by DeepPolyR at [1, 4]=[0.000000, 2.406402]
(Overall) Proved 4 8 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 9 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 8 9 by DeepPolyR at [0, 4]=[-4.211204, -4.211204]
(DeepPolyR) Proved 4 8 9 by DeepPolyR at [1, 4]=[0.000000, 3.008003]
(Overall) Proved 4 8 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 9 0 by DeepPolyR at [0, 4]=[-4.211204, 1.804802]
(DeepPolyR) Proved 4 9 0 by DeepPolyR at [0, 4]=[-4.211204, -3.008003]
(DeepPolyR) Proved 4 9 0 by DeepPolyR at [3, 4]=[-0.601601, 1.804802]
(Overall) Proved 4 9 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 9 1 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 9 1 by DeepPolyR at [0, 4]=[-3.609603, -0.601601]
(DeepPolyR) Proved 4 9 1 by DeepPolyR at [4, 4]=[3.609603, 3.609603]
(Overall) Proved 4 9 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 9 2 by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 9 2 by DeepPolyR at [0, 4]=[-3.008003, 0.601601]
(DeepPolyR) Proved 4 9 2 by DeepPolyR at [2, 4]=[1.203201, 4.211204]
(Overall) Proved 4 9 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 9 3 by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 9 3 by DeepPolyR at [0, 4]=[-3.008003, -0.601601]
(DeepPolyR) Fail to prove 4 9 3 by DeepPolyR at [4, 4]=[4.211204, 4.211204]
(Divide and Counter) Proved 4 9 3 by DeepPolyR at [0, 3]=[-3.008003, -0.601601]
(Divide and Counter) Fail to prove 4 9 3 by DeepPoly at [4]=4.211204
(Overall) Fail to prove 4 9 3 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 4 9 4 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 9 4 by DeepPolyR at [0, 4]=[-4.812805, -1.203201]
(DeepPolyR) Proved 4 9 4 by DeepPolyR at [4, 4]=[2.406402, 2.406402]
(Overall) Proved 4 9 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 9 5 by DeepPolyR at [0, 4]=[-3.609603, 3.609603]
(DeepPolyR) Proved 4 9 5 by DeepPolyR at [0, 4]=[-3.609603, 0.000000]
(DeepPolyR) Proved 4 9 5 by DeepPolyR at [2, 4]=[1.203201, 3.609603]
(Overall) Proved 4 9 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 9 6 by DeepPolyR at [0, 4]=[-3.008003, 4.211204]
(DeepPolyR) Proved 4 9 6 by DeepPolyR at [0, 4]=[-3.008003, -0.601601]
(DeepPolyR) Proved 4 9 6 by DeepPolyR at [4, 4]=[4.211204, 4.211204]
(Overall) Proved 4 9 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 9 7 by DeepPolyR at [0, 4]=[-4.812805, 2.406402]
(DeepPolyR) Proved 4 9 7 by DeepPolyR at [0, 4]=[-4.812805, -1.203201]
(DeepPolyR) Proved 4 9 7 by DeepPolyR at [4, 4]=[2.406402, 2.406402]
(Overall) Proved 4 9 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 9 8 by DeepPolyR at [0, 4]=[-4.211204, 3.008003]
(DeepPolyR) Proved 4 9 8 by DeepPolyR at [0, 4]=[-4.211204, -1.804802]
(DeepPolyR) Proved 4 9 8 by DeepPolyR at [3, 4]=[-0.601601, 3.008003]
(Overall) Proved 4 9 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 9 9 by DeepPolyR at [0, 4]=[-4.812805, 1.203201]
(DeepPolyR) Proved 4 9 9 by DeepPolyR at [0, 4]=[-4.812805, -3.008003]
(DeepPolyR) Proved 4 9 9 by DeepPolyR at [3, 4]=[-1.203201, 1.203201]
(Overall) Proved 4 9 9 with all masks. Summary: 1 1 1 

Elapsed Time: 2.641
flg =1
all_proved =0
queries = 351
