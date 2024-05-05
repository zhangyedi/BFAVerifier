GPUPoly 0.13.0S Debug (built Apr 26 2024 13:58:03) - Copyright (C) 2020 Department of Computer Science, ETH Zurich.
This program comes with ABSOLUTELY NO WARRANTY. This is free software, and you are welcome to redistribute it and to modify it under the terms of the GNU LGPLv3.

Warning: bit-flip at the first layer is not supported due to heavy technical debt (CPU version is supported)
Validating Bias
weights_path: benchmark/benchmark_QAT/QAT_mnist_3blk_10_10_10_qu_8.h5
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
label is 4
bit_all is 8
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
(Divide and Counter) Proved 2 0 0 by DeepPolyR at [0, 8]=[-0.594059, 1.103252]
(DeepPolyR) Proved 2 0 0 by DeepPolyR at [0, 8]=[-0.594059, 0.084866]
(DeepPolyR) Proved 2 0 0 by DeepPolyR at [2, 8]=[0.678924, 1.103252]
(Overall) Proved 2 0 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 0 1 by DeepPolyR at [0, 8]=[-1.018387, 1.018387]
(DeepPolyR) Proved 2 0 1 by DeepPolyR at [0, 8]=[-1.018387, -0.169731]
(DeepPolyR) Proved 2 0 1 by DeepPolyR at [8, 8]=[1.018387, 1.018387]
(Overall) Proved 2 0 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 0 2 by DeepPolyR at [0, 8]=[-0.848656, 1.188118]
(DeepPolyR) Proved 2 0 2 by DeepPolyR at [0, 8]=[-0.848656, -0.848656]
(DeepPolyR) Proved 2 0 2 by DeepPolyR at [1, 8]=[0.169731, 1.188118]
(Overall) Proved 2 0 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 0 3 by DeepPolyR at [0, 8]=[-0.795615, 1.241159]
(DeepPolyR) Proved 2 0 3 by DeepPolyR at [0, 8]=[-0.795615, -0.795615]
(DeepPolyR) Proved 2 0 3 by DeepPolyR at [1, 8]=[0.222772, 1.241159]
(Overall) Proved 2 0 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 0 4 by DeepPolyR at [0, 8]=[-0.838047, 1.198726]
(DeepPolyR) Proved 2 0 4 by DeepPolyR at [0, 8]=[-0.838047, -0.074257]
(DeepPolyR) Proved 2 0 4 by DeepPolyR at [8, 8]=[1.198726, 1.198726]
(Overall) Proved 2 0 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 0 5 by DeepPolyR at [0, 8]=[-0.785006, 1.251767]
(DeepPolyR) Proved 2 0 5 by DeepPolyR at [0, 8]=[-0.785006, 0.233380]
(DeepPolyR) Proved 2 0 5 by DeepPolyR at [2, 8]=[0.403111, 1.251767]
(Overall) Proved 2 0 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 0 6 by DeepPolyR at [0, 8]=[-0.010608, 1.347241]
(DeepPolyR) Proved 2 0 6 by DeepPolyR at [0, 8]=[-0.010608, 0.668316]
(DeepPolyR) Proved 2 0 6 by DeepPolyR at [2, 8]=[1.007778, 1.347241]
(Overall) Proved 2 0 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 0 7 by DeepPolyR at [0, 8]=[-0.944129, 1.092644]
(DeepPolyR) Proved 2 0 7 by DeepPolyR at [0, 8]=[-0.944129, 0.074257]
(DeepPolyR) Proved 2 0 7 by DeepPolyR at [2, 8]=[0.371287, 1.092644]
(Overall) Proved 2 0 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 0 8 by DeepPolyR at [0, 8]=[-1.007778, 1.028995]
(DeepPolyR) Proved 2 0 8 by DeepPolyR at [0, 8]=[-1.007778, -0.159123]
(DeepPolyR) Proved 2 0 8 by DeepPolyR at [8, 8]=[1.028995, 1.028995]
(Overall) Proved 2 0 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 0 9 by DeepPolyR at [0, 8]=[-1.357849, 0.678924]
(DeepPolyR) Proved 2 0 9 by DeepPolyR at [0, 8]=[-1.357849, -1.357849]
(DeepPolyR) Proved 2 0 9 by DeepPolyR at [1, 8]=[0.000000, 0.678924]
(Overall) Proved 2 0 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 1 0 by DeepPolyR at [0, 8]=[-1.209334, 0.827439]
(DeepPolyR) Proved 2 1 0 by DeepPolyR at [0, 8]=[-1.209334, -0.190947]
(DeepPolyR) Proved 2 1 0 by DeepPolyR at [8, 8]=[0.827439, 0.827439]
(Overall) Proved 2 1 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 1 1 by DeepPolyR at [0, 8]=[-1.251767, 0.275813]
(DeepPolyR) Proved 2 1 1 by DeepPolyR at [0, 8]=[-1.251767, -0.742574]
(DeepPolyR) Proved 2 1 1 by DeepPolyR at [7, 8]=[-0.403111, 0.275813]
(Overall) Proved 2 1 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 1 2 by DeepPolyR at [0, 8]=[-0.848656, 1.188118]
(DeepPolyR) Proved 2 1 2 by DeepPolyR at [0, 8]=[-0.848656, -0.848656]
(DeepPolyR) Proved 2 1 2 by DeepPolyR at [1, 8]=[0.169731, 1.188118]
(Overall) Proved 2 1 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 2 1 3 by DeepPolyR at [0, 8]=[-0.774398, 1.262375]
(DeepPolyR) Fail to prove 2 1 3 by DeepPolyR at [0, 8]=[-0.774398, 0.243988]
(DeepPolyR) Proved 2 1 3 by DeepPolyR at [2, 8]=[0.413720, 1.262375]
(Divide and Counter) Fail to prove 2 1 3 by DeepPolyR at [0, 1]=[-0.774398, 0.243988]
(Divide and Counter) Fail to prove 2 1 3 by DeepPoly at [0]=-0.774398
(Overall) Fail to prove 2 1 3 with all masks. Summary: 0 0 1 

(Divide and Counter) Proved 2 1 4 by DeepPolyR at [0, 8]=[-0.965346, 1.071428]
(DeepPolyR) Proved 2 1 4 by DeepPolyR at [0, 8]=[-0.965346, -0.116690]
(DeepPolyR) Proved 2 1 4 by DeepPolyR at [8, 8]=[1.071428, 1.071428]
(Overall) Proved 2 1 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 1 5 by DeepPolyR at [0, 8]=[-0.562234, 1.135077]
(DeepPolyR) Proved 2 1 5 by DeepPolyR at [0, 8]=[-0.562234, 0.116690]
(DeepPolyR) Proved 2 1 5 by DeepPolyR at [2, 8]=[0.710749, 1.135077]
(Overall) Proved 2 1 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 2 1 6 by DeepPolyR at [0, 8]=[-0.307638, 1.219942]
(DeepPolyR) Fail to prove 2 1 6 by DeepPolyR at [0, 8]=[-0.307638, 0.371287]
(DeepPolyR) Proved 2 1 6 by DeepPolyR at [2, 8]=[0.710749, 1.219942]
(Divide and Counter) Fail to prove 2 1 6 by DeepPolyR at [0, 1]=[-0.307638, 0.371287]
(Divide and Counter) Fail to prove 2 1 6 by DeepPoly at [0]=-0.307638
(Overall) Fail to prove 2 1 6 with all masks. Summary: 0 0 1 

(Divide and Counter) Proved 2 1 7 by DeepPolyR at [0, 8]=[-0.965346, 1.071428]
(DeepPolyR) Proved 2 1 7 by DeepPolyR at [0, 8]=[-0.965346, 0.053041]
(DeepPolyR) Proved 2 1 7 by DeepPolyR at [2, 8]=[0.350070, 1.071428]
(Overall) Proved 2 1 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 1 8 by DeepPolyR at [0, 8]=[-1.326024, 0.710749]
(DeepPolyR) Proved 2 1 8 by DeepPolyR at [0, 8]=[-1.326024, -1.326024]
(DeepPolyR) Proved 2 1 8 by DeepPolyR at [1, 8]=[0.010608, 0.710749]
(Overall) Proved 2 1 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 1 9 by DeepPolyR at [0, 8]=[-0.997170, 1.039603]
(DeepPolyR) Proved 2 1 9 by DeepPolyR at [0, 8]=[-0.997170, 0.021216]
(DeepPolyR) Proved 2 1 9 by DeepPolyR at [2, 8]=[0.339462, 1.039603]
(Overall) Proved 2 1 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 0 by DeepPolyR at [0, 8]=[-0.827439, 1.209334]
(DeepPolyR) Proved 2 2 0 by DeepPolyR at [0, 8]=[-0.827439, -0.827439]
(DeepPolyR) Proved 2 2 0 by DeepPolyR at [1, 8]=[0.190947, 1.209334]
(Overall) Proved 2 2 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 1 by DeepPolyR at [0, 8]=[-0.498585, 1.198726]
(DeepPolyR) Proved 2 2 1 by DeepPolyR at [0, 8]=[-0.498585, 0.180339]
(DeepPolyR) Proved 2 2 1 by DeepPolyR at [2, 8]=[0.689533, 1.198726]
(Overall) Proved 2 2 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 2 by DeepPolyR at [0, 8]=[-1.198726, 0.838047]
(DeepPolyR) Proved 2 2 2 by DeepPolyR at [0, 8]=[-1.198726, -0.180339]
(DeepPolyR) Proved 2 2 2 by DeepPolyR at [8, 8]=[0.838047, 0.838047]
(Overall) Proved 2 2 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 3 by DeepPolyR at [0, 8]=[-1.315416, 0.721357]
(DeepPolyR) Proved 2 2 3 by DeepPolyR at [0, 8]=[-1.315416, -0.297029]
(DeepPolyR) Proved 2 2 3 by DeepPolyR at [8, 8]=[0.721357, 0.721357]
(Overall) Proved 2 2 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 4 by DeepPolyR at [0, 8]=[-0.785006, 1.251767]
(DeepPolyR) Proved 2 2 4 by DeepPolyR at [0, 8]=[-0.785006, 0.233380]
(DeepPolyR) Proved 2 2 4 by DeepPolyR at [2, 8]=[0.403111, 1.251767]
(Overall) Proved 2 2 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 5 by DeepPolyR at [0, 8]=[-1.156293, 0.541018]
(DeepPolyR) Proved 2 2 5 by DeepPolyR at [0, 8]=[-1.156293, -0.731965]
(DeepPolyR) Proved 2 2 5 by DeepPolyR at [7, 8]=[-0.137907, 0.541018]
(Overall) Proved 2 2 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 6 by DeepPolyR at [0, 8]=[-1.357849, 0.678924]
(DeepPolyR) Proved 2 2 6 by DeepPolyR at [0, 8]=[-1.357849, -1.357849]
(DeepPolyR) Proved 2 2 6 by DeepPolyR at [1, 8]=[0.000000, 0.678924]
(Overall) Proved 2 2 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 7 by DeepPolyR at [0, 8]=[-1.304808, 0.731965]
(DeepPolyR) Proved 2 2 7 by DeepPolyR at [0, 8]=[-1.304808, -0.286421]
(DeepPolyR) Proved 2 2 7 by DeepPolyR at [8, 8]=[0.731965, 0.731965]
(Overall) Proved 2 2 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 8 by DeepPolyR at [0, 8]=[-1.230551, 0.806223]
(DeepPolyR) Proved 2 2 8 by DeepPolyR at [0, 8]=[-1.230551, -0.212164]
(DeepPolyR) Proved 2 2 8 by DeepPolyR at [8, 8]=[0.806223, 0.806223]
(Overall) Proved 2 2 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 9 by DeepPolyR at [0, 8]=[-1.230551, 0.806223]
(DeepPolyR) Proved 2 2 9 by DeepPolyR at [0, 8]=[-1.230551, -0.212164]
(DeepPolyR) Proved 2 2 9 by DeepPolyR at [8, 8]=[0.806223, 0.806223]
(Overall) Proved 2 2 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 0 by DeepPolyR at [0, 8]=[-1.124469, 0.912305]
(DeepPolyR) Proved 2 3 0 by DeepPolyR at [0, 8]=[-1.124469, -1.124469]
(DeepPolyR) Proved 2 3 0 by DeepPolyR at [1, 8]=[0.063649, 0.912305]
(Overall) Proved 2 3 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 1 by DeepPolyR at [0, 8]=[-0.169731, 1.272983]
(DeepPolyR) Proved 2 3 1 by DeepPolyR at [0, 8]=[-0.169731, 0.509193]
(DeepPolyR) Proved 2 3 1 by DeepPolyR at [2, 8]=[0.848656, 1.272983]
(Overall) Proved 2 3 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 2 by DeepPolyR at [0, 8]=[-1.198726, 0.838047]
(DeepPolyR) Proved 2 3 2 by DeepPolyR at [0, 8]=[-1.198726, -1.198726]
(DeepPolyR) Proved 2 3 2 by DeepPolyR at [1, 8]=[0.074257, 0.838047]
(Overall) Proved 2 3 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 3 by DeepPolyR at [0, 8]=[-1.166901, 0.530410]
(DeepPolyR) Proved 2 3 3 by DeepPolyR at [0, 8]=[-1.166901, -0.742574]
(DeepPolyR) Proved 2 3 3 by DeepPolyR at [7, 8]=[-0.148515, 0.530410]
(Overall) Proved 2 3 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 4 by DeepPolyR at [0, 8]=[-0.774398, 1.262375]
(DeepPolyR) Proved 2 3 4 by DeepPolyR at [0, 8]=[-0.774398, 0.243988]
(DeepPolyR) Proved 2 3 4 by DeepPolyR at [2, 8]=[0.413720, 1.262375]
(Overall) Proved 2 3 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 5 by DeepPolyR at [0, 8]=[-0.997170, 1.039603]
(DeepPolyR) Proved 2 3 5 by DeepPolyR at [0, 8]=[-0.997170, -0.148515]
(DeepPolyR) Proved 2 3 5 by DeepPolyR at [8, 8]=[1.039603, 1.039603]
(Overall) Proved 2 3 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 6 by DeepPolyR at [0, 8]=[-1.283592, 0.243988]
(DeepPolyR) Proved 2 3 6 by DeepPolyR at [0, 8]=[-1.283592, -0.774398]
(DeepPolyR) Proved 2 3 6 by DeepPolyR at [7, 8]=[-0.434936, 0.243988]
(Overall) Proved 2 3 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 7 by DeepPolyR at [0, 8]=[-0.816831, 1.219942]
(DeepPolyR) Proved 2 3 7 by DeepPolyR at [0, 8]=[-0.816831, -0.053041]
(DeepPolyR) Proved 2 3 7 by DeepPolyR at [8, 8]=[1.219942, 1.219942]
(Overall) Proved 2 3 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 8 by DeepPolyR at [0, 8]=[-0.806223, 1.230551]
(DeepPolyR) Proved 2 3 8 by DeepPolyR at [0, 8]=[-0.806223, -0.806223]
(DeepPolyR) Proved 2 3 8 by DeepPolyR at [1, 8]=[0.212164, 1.230551]
(Overall) Proved 2 3 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 9 by DeepPolyR at [0, 8]=[-0.710749, 1.326024]
(DeepPolyR) Proved 2 3 9 by DeepPolyR at [0, 8]=[-0.710749, 0.307638]
(DeepPolyR) Proved 2 3 9 by DeepPolyR at [2, 8]=[0.477369, 1.326024]
(Overall) Proved 2 3 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 0 by DeepPolyR at [0, 8]=[-0.753182, 1.283592]
(DeepPolyR) Proved 2 4 0 by DeepPolyR at [0, 8]=[-0.753182, -0.031825]
(DeepPolyR) Proved 2 4 0 by DeepPolyR at [8, 8]=[1.283592, 1.283592]
(Overall) Proved 2 4 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 1 by DeepPolyR at [0, 8]=[-1.347241, 0.689533]
(DeepPolyR) Proved 2 4 1 by DeepPolyR at [0, 8]=[-1.347241, -1.347241]
(DeepPolyR) Proved 2 4 1 by DeepPolyR at [1, 8]=[0.000000, 0.689533]
(Overall) Proved 2 4 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 2 by DeepPolyR at [0, 8]=[-0.434936, 1.262375]
(DeepPolyR) Proved 2 4 2 by DeepPolyR at [0, 8]=[-0.434936, 0.243988]
(DeepPolyR) Proved 2 4 2 by DeepPolyR at [2, 8]=[0.753182, 1.262375]
(Overall) Proved 2 4 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 3 by DeepPolyR at [0, 8]=[-1.304808, 0.731965]
(DeepPolyR) Proved 2 4 3 by DeepPolyR at [0, 8]=[-1.304808, -1.304808]
(DeepPolyR) Proved 2 4 3 by DeepPolyR at [1, 8]=[0.010608, 0.731965]
(Overall) Proved 2 4 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 4 by DeepPolyR at [0, 8]=[-1.166901, 0.869872]
(DeepPolyR) Proved 2 4 4 by DeepPolyR at [0, 8]=[-1.166901, -1.166901]
(DeepPolyR) Proved 2 4 4 by DeepPolyR at [1, 8]=[0.021216, 0.869872]
(Overall) Proved 2 4 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 5 by DeepPolyR at [0, 8]=[-0.763790, 1.272983]
(DeepPolyR) Proved 2 4 5 by DeepPolyR at [0, 8]=[-0.763790, -0.763790]
(DeepPolyR) Proved 2 4 5 by DeepPolyR at [1, 8]=[0.254597, 1.272983]
(Overall) Proved 2 4 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 6 by DeepPolyR at [0, 8]=[-1.082036, 0.954737]
(DeepPolyR) Proved 2 4 6 by DeepPolyR at [0, 8]=[-1.082036, -0.360679]
(DeepPolyR) Proved 2 4 6 by DeepPolyR at [7, 8]=[-0.063649, 0.954737]
(Overall) Proved 2 4 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 7 by DeepPolyR at [0, 8]=[-0.148515, 1.294200]
(DeepPolyR) Proved 2 4 7 by DeepPolyR at [0, 8]=[-0.148515, 0.530410]
(DeepPolyR) Proved 2 4 7 by DeepPolyR at [2, 8]=[0.869872, 1.294200]
(Overall) Proved 2 4 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 8 by DeepPolyR at [0, 8]=[-0.647100, 1.050211]
(DeepPolyR) Proved 2 4 8 by DeepPolyR at [0, 8]=[-0.647100, 0.031825]
(DeepPolyR) Proved 2 4 8 by DeepPolyR at [2, 8]=[0.689533, 1.050211]
(Overall) Proved 2 4 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 9 by DeepPolyR at [0, 8]=[-0.159123, 1.283592]
(DeepPolyR) Proved 2 4 9 by DeepPolyR at [0, 8]=[-0.159123, 0.519802]
(DeepPolyR) Proved 2 4 9 by DeepPolyR at [2, 8]=[0.859264, 1.283592]
(Overall) Proved 2 4 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 0 by DeepPolyR at [0, 8]=[-0.572842, 1.124469]
(DeepPolyR) Proved 2 5 0 by DeepPolyR at [0, 8]=[-0.572842, 0.106082]
(DeepPolyR) Proved 2 5 0 by DeepPolyR at [2, 8]=[0.700141, 1.124469]
(Overall) Proved 2 5 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 1 by DeepPolyR at [0, 8]=[-0.944129, 1.092644]
(DeepPolyR) Proved 2 5 1 by DeepPolyR at [0, 8]=[-0.944129, 0.074257]
(DeepPolyR) Proved 2 5 1 by DeepPolyR at [2, 8]=[0.371287, 1.092644]
(Overall) Proved 2 5 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 2 by DeepPolyR at [0, 8]=[-1.283592, 0.413720]
(DeepPolyR) Proved 2 5 2 by DeepPolyR at [0, 8]=[-1.283592, -0.774398]
(DeepPolyR) Proved 2 5 2 by DeepPolyR at [7, 8]=[-0.265205, 0.413720]
(Overall) Proved 2 5 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 3 by DeepPolyR at [0, 8]=[-1.039603, 0.997170]
(DeepPolyR) Proved 2 5 3 by DeepPolyR at [0, 8]=[-1.039603, -1.039603]
(DeepPolyR) Proved 2 5 3 by DeepPolyR at [1, 8]=[0.148515, 0.997170]
(Overall) Proved 2 5 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 4 by DeepPolyR at [0, 8]=[-0.774398, 1.262375]
(DeepPolyR) Proved 2 5 4 by DeepPolyR at [0, 8]=[-0.774398, 0.243988]
(DeepPolyR) Proved 2 5 4 by DeepPolyR at [2, 8]=[0.413720, 1.262375]
(Overall) Proved 2 5 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 5 by DeepPolyR at [0, 8]=[-1.304808, 0.731965]
(DeepPolyR) Proved 2 5 5 by DeepPolyR at [0, 8]=[-1.304808, -0.286421]
(DeepPolyR) Proved 2 5 5 by DeepPolyR at [8, 8]=[0.731965, 0.731965]
(Overall) Proved 2 5 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 6 by DeepPolyR at [0, 8]=[-0.678924, 1.018387]
(DeepPolyR) Proved 2 5 6 by DeepPolyR at [0, 8]=[-0.678924, 0.000000]
(DeepPolyR) Proved 2 5 6 by DeepPolyR at [2, 8]=[0.678924, 1.018387]
(Overall) Proved 2 5 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 7 by DeepPolyR at [0, 8]=[-1.241159, 0.795615]
(DeepPolyR) Proved 2 5 7 by DeepPolyR at [0, 8]=[-1.241159, -0.222772]
(DeepPolyR) Proved 2 5 7 by DeepPolyR at [8, 8]=[0.795615, 0.795615]
(Overall) Proved 2 5 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 8 by DeepPolyR at [0, 8]=[-1.283592, 0.243988]
(DeepPolyR) Proved 2 5 8 by DeepPolyR at [0, 8]=[-1.283592, -0.774398]
(DeepPolyR) Proved 2 5 8 by DeepPolyR at [7, 8]=[-0.434936, 0.243988]
(Overall) Proved 2 5 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 9 by DeepPolyR at [0, 8]=[-1.336632, 0.190947]
(DeepPolyR) Proved 2 5 9 by DeepPolyR at [0, 8]=[-1.336632, -0.827439]
(DeepPolyR) Proved 2 5 9 by DeepPolyR at [7, 8]=[-0.487977, 0.190947]
(Overall) Proved 2 5 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 0 by DeepPolyR at [0, 8]=[-0.859264, 1.177510]
(DeepPolyR) Proved 2 6 0 by DeepPolyR at [0, 8]=[-0.859264, 0.159123]
(DeepPolyR) Proved 2 6 0 by DeepPolyR at [2, 8]=[0.413720, 1.177510]
(Overall) Proved 2 6 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 1 by DeepPolyR at [0, 8]=[-0.922913, 1.113860]
(DeepPolyR) Proved 2 6 1 by DeepPolyR at [0, 8]=[-0.922913, -0.074257]
(DeepPolyR) Proved 2 6 1 by DeepPolyR at [8, 8]=[1.113860, 1.113860]
(Overall) Proved 2 6 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 2 by DeepPolyR at [0, 8]=[-1.304808, 0.731965]
(DeepPolyR) Proved 2 6 2 by DeepPolyR at [0, 8]=[-1.304808, -0.286421]
(DeepPolyR) Proved 2 6 2 by DeepPolyR at [8, 8]=[0.731965, 0.731965]
(Overall) Proved 2 6 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 3 by DeepPolyR at [0, 8]=[-0.360679, 1.336632]
(DeepPolyR) Proved 2 6 3 by DeepPolyR at [0, 8]=[-0.360679, 0.318246]
(DeepPolyR) Proved 2 6 3 by DeepPolyR at [2, 8]=[0.827439, 1.336632]
(Overall) Proved 2 6 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 4 by DeepPolyR at [0, 8]=[-0.806223, 1.230551]
(DeepPolyR) Proved 2 6 4 by DeepPolyR at [0, 8]=[-0.806223, -0.806223]
(DeepPolyR) Proved 2 6 4 by DeepPolyR at [1, 8]=[0.212164, 1.230551]
(Overall) Proved 2 6 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 5 by DeepPolyR at [0, 8]=[-1.241159, 0.795615]
(DeepPolyR) Proved 2 6 5 by DeepPolyR at [0, 8]=[-1.241159, -0.222772]
(DeepPolyR) Proved 2 6 5 by DeepPolyR at [8, 8]=[0.795615, 0.795615]
(Overall) Proved 2 6 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 2 6 6 by DeepPolyR at [0, 8]=[-1.145685, 0.891088]
(DeepPolyR) Proved 2 6 6 by DeepPolyR at [0, 8]=[-1.145685, -1.145685]
(DeepPolyR) Fail to prove 2 6 6 by DeepPolyR at [1, 8]=[0.042433, 0.891088]
(Divide and Counter) Proved 2 6 6 by DeepPoly at [0]=-1.145685
(Divide and Counter) Fail to prove 2 6 6 by DeepPolyR at [1, 8]=[0.042433, 0.891088]
(Divide and Counter) Proved 2 6 6 by DeepPolyR at [1, 6]=[0.042433, 0.297029]
(Divide and Counter) Fail to prove 2 6 6 by DeepPolyR at [7, 8]=[0.551626, 0.891088]
(Divide and Counter) Proved 2 6 6 by DeepPoly at [7]=0.551626
(Divide and Counter) Fail to prove 2 6 6 by DeepPoly at [8]=0.891088
(Overall) Fail to prove 2 6 6 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 2 6 7 by DeepPolyR at [0, 8]=[-1.145685, 0.891088]
(DeepPolyR) Proved 2 6 7 by DeepPolyR at [0, 8]=[-1.145685, -0.381895]
(DeepPolyR) Proved 2 6 7 by DeepPolyR at [7, 8]=[-0.127298, 0.891088]
(Overall) Proved 2 6 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 8 by DeepPolyR at [0, 8]=[-1.262375, 0.265205]
(DeepPolyR) Proved 2 6 8 by DeepPolyR at [0, 8]=[-1.262375, -0.753182]
(DeepPolyR) Proved 2 6 8 by DeepPolyR at [7, 8]=[-0.413720, 0.265205]
(Overall) Proved 2 6 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 9 by DeepPolyR at [0, 8]=[-1.336632, 0.700141]
(DeepPolyR) Proved 2 6 9 by DeepPolyR at [0, 8]=[-1.336632, -0.318246]
(DeepPolyR) Proved 2 6 9 by DeepPolyR at [8, 8]=[0.700141, 0.700141]
(Overall) Proved 2 6 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 0 by DeepPolyR at [0, 8]=[-0.880480, 1.156293]
(DeepPolyR) Proved 2 7 0 by DeepPolyR at [0, 8]=[-0.880480, 0.137907]
(DeepPolyR) Proved 2 7 0 by DeepPolyR at [2, 8]=[0.392503, 1.156293]
(Overall) Proved 2 7 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 1 by DeepPolyR at [0, 8]=[-0.880480, 1.156293]
(DeepPolyR) Proved 2 7 1 by DeepPolyR at [0, 8]=[-0.880480, 0.137907]
(DeepPolyR) Proved 2 7 1 by DeepPolyR at [2, 8]=[0.392503, 1.156293]
(Overall) Proved 2 7 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 2 by DeepPolyR at [0, 8]=[-0.625883, 1.071428]
(DeepPolyR) Proved 2 7 2 by DeepPolyR at [0, 8]=[-0.625883, 0.053041]
(DeepPolyR) Proved 2 7 2 by DeepPolyR at [2, 8]=[0.689533, 1.071428]
(Overall) Proved 2 7 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 3 by DeepPolyR at [0, 8]=[-1.039603, 0.657708]
(DeepPolyR) Proved 2 7 3 by DeepPolyR at [0, 8]=[-1.039603, -0.689533]
(DeepPolyR) Proved 2 7 3 by DeepPolyR at [7, 8]=[-0.021216, 0.657708]
(Overall) Proved 2 7 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 4 by DeepPolyR at [0, 8]=[-0.848656, 1.188118]
(DeepPolyR) Proved 2 7 4 by DeepPolyR at [0, 8]=[-0.848656, -0.848656]
(DeepPolyR) Proved 2 7 4 by DeepPolyR at [1, 8]=[0.169731, 1.188118]
(Overall) Proved 2 7 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 5 by DeepPolyR at [0, 8]=[-0.572842, 1.124469]
(DeepPolyR) Proved 2 7 5 by DeepPolyR at [0, 8]=[-0.572842, 0.106082]
(DeepPolyR) Proved 2 7 5 by DeepPolyR at [2, 8]=[0.700141, 1.124469]
(Overall) Proved 2 7 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 6 by DeepPolyR at [0, 8]=[-1.347241, 0.689533]
(DeepPolyR) Proved 2 7 6 by DeepPolyR at [0, 8]=[-1.347241, -0.328854]
(DeepPolyR) Proved 2 7 6 by DeepPolyR at [8, 8]=[0.689533, 0.689533]
(Overall) Proved 2 7 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 7 by DeepPolyR at [0, 8]=[-0.848656, 1.188118]
(DeepPolyR) Proved 2 7 7 by DeepPolyR at [0, 8]=[-0.848656, -0.848656]
(DeepPolyR) Proved 2 7 7 by DeepPolyR at [1, 8]=[0.169731, 1.188118]
(Overall) Proved 2 7 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 8 by DeepPolyR at [0, 8]=[-0.572842, 1.124469]
(DeepPolyR) Proved 2 7 8 by DeepPolyR at [0, 8]=[-0.572842, 0.106082]
(DeepPolyR) Proved 2 7 8 by DeepPolyR at [2, 8]=[0.700141, 1.124469]
(Overall) Proved 2 7 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 9 by DeepPolyR at [0, 8]=[-0.763790, 1.272983]
(DeepPolyR) Proved 2 7 9 by DeepPolyR at [0, 8]=[-0.763790, -0.763790]
(DeepPolyR) Proved 2 7 9 by DeepPolyR at [1, 8]=[0.254597, 1.272983]
(Overall) Proved 2 7 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 0 by DeepPolyR at [0, 8]=[-0.880480, 1.156293]
(DeepPolyR) Proved 2 8 0 by DeepPolyR at [0, 8]=[-0.880480, 0.137907]
(DeepPolyR) Proved 2 8 0 by DeepPolyR at [2, 8]=[0.392503, 1.156293]
(Overall) Proved 2 8 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 1 by DeepPolyR at [0, 8]=[-1.272983, 0.763790]
(DeepPolyR) Proved 2 8 1 by DeepPolyR at [0, 8]=[-1.272983, -0.254597]
(DeepPolyR) Proved 2 8 1 by DeepPolyR at [8, 8]=[0.763790, 0.763790]
(Overall) Proved 2 8 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 2 by DeepPolyR at [0, 8]=[-0.975954, 1.060819]
(DeepPolyR) Proved 2 8 2 by DeepPolyR at [0, 8]=[-0.975954, -0.127298]
(DeepPolyR) Proved 2 8 2 by DeepPolyR at [8, 8]=[1.060819, 1.060819]
(Overall) Proved 2 8 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 3 by DeepPolyR at [0, 8]=[-0.806223, 1.230551]
(DeepPolyR) Proved 2 8 3 by DeepPolyR at [0, 8]=[-0.806223, -0.806223]
(DeepPolyR) Proved 2 8 3 by DeepPolyR at [1, 8]=[0.212164, 1.230551]
(Overall) Proved 2 8 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 4 by DeepPolyR at [0, 8]=[-1.209334, 0.827439]
(DeepPolyR) Proved 2 8 4 by DeepPolyR at [0, 8]=[-1.209334, -1.209334]
(DeepPolyR) Proved 2 8 4 by DeepPolyR at [1, 8]=[0.063649, 0.827439]
(Overall) Proved 2 8 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 5 by DeepPolyR at [0, 8]=[-0.869872, 1.166901]
(DeepPolyR) Proved 2 8 5 by DeepPolyR at [0, 8]=[-0.869872, -0.021216]
(DeepPolyR) Proved 2 8 5 by DeepPolyR at [8, 8]=[1.166901, 1.166901]
(Overall) Proved 2 8 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 6 by DeepPolyR at [0, 8]=[-0.328854, 1.198726]
(DeepPolyR) Proved 2 8 6 by DeepPolyR at [0, 8]=[-0.328854, 0.350070]
(DeepPolyR) Proved 2 8 6 by DeepPolyR at [2, 8]=[0.689533, 1.198726]
(Overall) Proved 2 8 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 7 by DeepPolyR at [0, 8]=[-1.039603, 0.997170]
(DeepPolyR) Proved 2 8 7 by DeepPolyR at [0, 8]=[-1.039603, -0.350070]
(DeepPolyR) Proved 2 8 7 by DeepPolyR at [7, 8]=[-0.021216, 0.997170]
(Overall) Proved 2 8 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 8 by DeepPolyR at [0, 8]=[-1.251767, 0.445544]
(DeepPolyR) Proved 2 8 8 by DeepPolyR at [0, 8]=[-1.251767, -0.742574]
(DeepPolyR) Proved 2 8 8 by DeepPolyR at [7, 8]=[-0.233380, 0.445544]
(Overall) Proved 2 8 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 9 by DeepPolyR at [0, 8]=[-0.944129, 1.092644]
(DeepPolyR) Proved 2 8 9 by DeepPolyR at [0, 8]=[-0.944129, -0.095474]
(DeepPolyR) Proved 2 8 9 by DeepPolyR at [8, 8]=[1.092644, 1.092644]
(Overall) Proved 2 8 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 0 by DeepPolyR at [0, 8]=[-1.060819, 0.975954]
(DeepPolyR) Proved 2 9 0 by DeepPolyR at [0, 8]=[-1.060819, -1.060819]
(DeepPolyR) Proved 2 9 0 by DeepPolyR at [1, 8]=[0.127298, 0.975954]
(Overall) Proved 2 9 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 1 by DeepPolyR at [0, 8]=[-0.254597, 1.272983]
(DeepPolyR) Proved 2 9 1 by DeepPolyR at [0, 8]=[-0.254597, 0.424328]
(DeepPolyR) Proved 2 9 1 by DeepPolyR at [2, 8]=[0.763790, 1.272983]
(Overall) Proved 2 9 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 2 by DeepPolyR at [0, 8]=[-1.198726, 0.838047]
(DeepPolyR) Proved 2 9 2 by DeepPolyR at [0, 8]=[-1.198726, -0.180339]
(DeepPolyR) Proved 2 9 2 by DeepPolyR at [8, 8]=[0.838047, 0.838047]
(Overall) Proved 2 9 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 3 by DeepPolyR at [0, 8]=[-1.326024, 0.371287]
(DeepPolyR) Proved 2 9 3 by DeepPolyR at [0, 8]=[-1.326024, -0.816831]
(DeepPolyR) Proved 2 9 3 by DeepPolyR at [7, 8]=[-0.307638, 0.371287]
(Overall) Proved 2 9 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 4 by DeepPolyR at [0, 8]=[-0.615275, 1.082036]
(DeepPolyR) Proved 2 9 4 by DeepPolyR at [0, 8]=[-0.615275, 0.063649]
(DeepPolyR) Proved 2 9 4 by DeepPolyR at [2, 8]=[0.700141, 1.082036]
(Overall) Proved 2 9 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 5 by DeepPolyR at [0, 8]=[-1.050211, 0.647100]
(DeepPolyR) Proved 2 9 5 by DeepPolyR at [0, 8]=[-1.050211, -0.689533]
(DeepPolyR) Proved 2 9 5 by DeepPolyR at [7, 8]=[-0.031825, 0.647100]
(Overall) Proved 2 9 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 6 by DeepPolyR at [0, 8]=[-1.177510, 0.519802]
(DeepPolyR) Proved 2 9 6 by DeepPolyR at [0, 8]=[-1.177510, -0.753182]
(DeepPolyR) Proved 2 9 6 by DeepPolyR at [7, 8]=[-0.159123, 0.519802]
(Overall) Proved 2 9 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 7 by DeepPolyR at [0, 8]=[-0.742574, 1.294200]
(DeepPolyR) Proved 2 9 7 by DeepPolyR at [0, 8]=[-0.742574, -0.021216]
(DeepPolyR) Proved 2 9 7 by DeepPolyR at [8, 8]=[1.294200, 1.294200]
(Overall) Proved 2 9 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 8 by DeepPolyR at [0, 8]=[-1.124469, 0.912305]
(DeepPolyR) Proved 2 9 8 by DeepPolyR at [0, 8]=[-1.124469, -1.124469]
(DeepPolyR) Proved 2 9 8 by DeepPolyR at [1, 8]=[0.063649, 0.912305]
(Overall) Proved 2 9 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 9 by DeepPolyR at [0, 8]=[-1.071428, 0.965346]
(DeepPolyR) Proved 2 9 9 by DeepPolyR at [0, 8]=[-1.071428, -1.071428]
(DeepPolyR) Proved 2 9 9 by DeepPolyR at [1, 8]=[0.116690, 0.965346]
(Overall) Proved 2 9 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 0 by DeepPolyR at [0, 8]=[-0.940394, 1.567323]
(DeepPolyR) Proved 3 0 0 by DeepPolyR at [0, 8]=[-0.940394, 0.313465]
(DeepPolyR) Proved 3 0 0 by DeepPolyR at [2, 8]=[0.522441, 1.567323]
(Overall) Proved 3 0 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 1 by DeepPolyR at [0, 8]=[-1.567323, 0.940394]
(DeepPolyR) Proved 3 0 1 by DeepPolyR at [0, 8]=[-1.567323, -1.567323]
(DeepPolyR) Proved 3 0 1 by DeepPolyR at [1, 8]=[0.000000, 0.940394]
(Overall) Proved 3 0 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 2 by DeepPolyR at [0, 8]=[-0.431014, 1.658750]
(DeepPolyR) Proved 3 0 2 by DeepPolyR at [0, 8]=[-0.431014, 0.404892]
(DeepPolyR) Proved 3 0 2 by DeepPolyR at [2, 8]=[1.031821, 1.658750]
(Overall) Proved 3 0 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 3 by DeepPolyR at [0, 8]=[-1.136309, 1.371408]
(DeepPolyR) Proved 3 0 3 by DeepPolyR at [0, 8]=[-1.136309, -0.091427]
(DeepPolyR) Proved 3 0 3 by DeepPolyR at [8, 8]=[1.371408, 1.371408]
(Overall) Proved 3 0 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 4 by DeepPolyR at [0, 8]=[-1.332225, 1.175492]
(DeepPolyR) Proved 3 0 4 by DeepPolyR at [0, 8]=[-1.332225, -0.078366]
(DeepPolyR) Proved 3 0 4 by DeepPolyR at [8, 8]=[1.175492, 1.175492]
(Overall) Proved 3 0 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 5 by DeepPolyR at [0, 8]=[-0.013061, 1.658750]
(DeepPolyR) Proved 3 0 5 by DeepPolyR at [0, 8]=[-0.013061, -0.013061]
(DeepPolyR) Proved 3 0 5 by DeepPolyR at [1, 8]=[0.822845, 1.658750]
(Overall) Proved 3 0 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 6 by DeepPolyR at [0, 8]=[-0.065305, 1.658750]
(DeepPolyR) Proved 3 0 6 by DeepPolyR at [0, 8]=[-0.065305, 0.770600]
(DeepPolyR) Proved 3 0 6 by DeepPolyR at [2, 8]=[1.188553, 1.658750]
(Overall) Proved 3 0 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 7 by DeepPolyR at [0, 8]=[-1.567323, 0.940394]
(DeepPolyR) Proved 3 0 7 by DeepPolyR at [0, 8]=[-1.567323, -1.567323]
(DeepPolyR) Proved 3 0 7 by DeepPolyR at [1, 8]=[0.000000, 0.940394]
(Overall) Proved 3 0 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 8 by DeepPolyR at [0, 8]=[-0.287343, 1.593445]
(DeepPolyR) Proved 3 0 8 by DeepPolyR at [0, 8]=[-0.287343, 0.548563]
(DeepPolyR) Proved 3 0 8 by DeepPolyR at [2, 8]=[0.966516, 1.593445]
(Overall) Proved 3 0 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 9 by DeepPolyR at [0, 8]=[-1.384469, 1.123248]
(DeepPolyR) Proved 3 0 9 by DeepPolyR at [0, 8]=[-1.384469, -1.384469]
(DeepPolyR) Proved 3 0 9 by DeepPolyR at [1, 8]=[0.078366, 1.123248]
(Overall) Proved 3 0 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 3 1 0 by DeepPolyR at [0, 8]=[-1.097126, 1.410591]
(DeepPolyR) Proved 3 1 0 by DeepPolyR at [0, 8]=[-1.097126, -0.052244]
(DeepPolyR) Fail to prove 3 1 0 by DeepPolyR at [8, 8]=[1.410591, 1.410591]
(Divide and Counter) Proved 3 1 0 by DeepPolyR at [0, 7]=[-1.097126, -0.052244]
(Divide and Counter) Fail to prove 3 1 0 by DeepPoly at [8]=1.410591
(Overall) Fail to prove 3 1 0 with all masks. Summary: 0 1 0 

(Divide and Counter) Fail to prove 3 1 1 by DeepPolyR at [0, 8]=[-1.371408, 0.718356]
(DeepPolyR) Proved 3 1 1 by DeepPolyR at [0, 8]=[-1.371408, -0.848967]
(DeepPolyR) Fail to prove 3 1 1 by DeepPolyR at [7, 8]=[-0.117549, 0.718356]
(Divide and Counter) Proved 3 1 1 by DeepPolyR at [0, 6]=[-1.371408, -0.848967]
(Divide and Counter) Fail to prove 3 1 1 by DeepPolyR at [7, 8]=[-0.117549, 0.718356]
(Divide and Counter) Proved 3 1 1 by DeepPoly at [7]=-0.117549
(Divide and Counter) Fail to prove 3 1 1 by DeepPoly at [8]=0.718356
(Overall) Fail to prove 3 1 1 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 3 1 2 by DeepPolyR at [0, 8]=[-0.444075, 1.645689]
(DeepPolyR) Proved 3 1 2 by DeepPolyR at [0, 8]=[-0.444075, 0.391831]
(DeepPolyR) Proved 3 1 2 by DeepPolyR at [2, 8]=[1.018760, 1.645689]
(Overall) Proved 3 1 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 3 by DeepPolyR at [0, 8]=[-1.097126, 1.410591]
(DeepPolyR) Proved 3 1 3 by DeepPolyR at [0, 8]=[-1.097126, -1.097126]
(DeepPolyR) Proved 3 1 3 by DeepPolyR at [1, 8]=[0.156732, 1.410591]
(Overall) Proved 3 1 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 4 by DeepPolyR at [0, 8]=[-0.901211, 1.606506]
(DeepPolyR) Proved 3 1 4 by DeepPolyR at [0, 8]=[-0.901211, -0.013061]
(DeepPolyR) Proved 3 1 4 by DeepPolyR at [8, 8]=[1.606506, 1.606506]
(Overall) Proved 3 1 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 5 by DeepPolyR at [0, 8]=[-0.378770, 1.502018]
(DeepPolyR) Proved 3 1 5 by DeepPolyR at [0, 8]=[-0.378770, 0.457136]
(DeepPolyR) Proved 3 1 5 by DeepPolyR at [2, 8]=[0.875089, 1.502018]
(Overall) Proved 3 1 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 6 by DeepPolyR at [0, 8]=[-0.417953, 1.462835]
(DeepPolyR) Proved 3 1 6 by DeepPolyR at [0, 8]=[-0.417953, 0.417953]
(DeepPolyR) Proved 3 1 6 by DeepPolyR at [2, 8]=[0.835906, 1.462835]
(Overall) Proved 3 1 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 7 by DeepPolyR at [0, 8]=[-1.541201, 0.966516]
(DeepPolyR) Proved 3 1 7 by DeepPolyR at [0, 8]=[-1.541201, -1.541201]
(DeepPolyR) Proved 3 1 7 by DeepPolyR at [1, 8]=[0.026122, 0.966516]
(Overall) Proved 3 1 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 8 by DeepPolyR at [0, 8]=[-1.502018, 1.005699]
(DeepPolyR) Proved 3 1 8 by DeepPolyR at [0, 8]=[-1.502018, -1.502018]
(DeepPolyR) Proved 3 1 8 by DeepPolyR at [1, 8]=[0.065305, 1.005699]
(Overall) Proved 3 1 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 9 by DeepPolyR at [0, 8]=[-0.587746, 1.502018]
(DeepPolyR) Proved 3 1 9 by DeepPolyR at [0, 8]=[-0.587746, 0.248159]
(DeepPolyR) Proved 3 1 9 by DeepPolyR at [2, 8]=[0.875089, 1.502018]
(Overall) Proved 3 1 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 2 0 by DeepPolyR at [0, 8]=[-1.279980, 1.227736]
(DeepPolyR) Proved 3 2 0 by DeepPolyR at [0, 8]=[-1.279980, -1.279980]
(DeepPolyR) Proved 3 2 0 by DeepPolyR at [1, 8]=[0.182854, 1.227736]
(Overall) Proved 3 2 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 2 1 by DeepPolyR at [0, 8]=[-0.208976, 1.567323]
(DeepPolyR) Proved 3 2 1 by DeepPolyR at [0, 8]=[-0.208976, 0.626929]
(DeepPolyR) Proved 3 2 1 by DeepPolyR at [2, 8]=[1.044882, 1.567323]
(Overall) Proved 3 2 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 2 2 by DeepPolyR at [0, 8]=[-1.475896, 1.031821]
(DeepPolyR) Proved 3 2 2 by DeepPolyR at [0, 8]=[-1.475896, -0.222037]
(DeepPolyR) Proved 3 2 2 by DeepPolyR at [8, 8]=[1.031821, 1.031821]
(Overall) Proved 3 2 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 2 3 by DeepPolyR at [0, 8]=[-1.528140, 0.561624]
(DeepPolyR) Proved 3 2 3 by DeepPolyR at [0, 8]=[-1.528140, -0.901211]
(DeepPolyR) Proved 3 2 3 by DeepPolyR at [7, 8]=[-0.274282, 0.561624]
(Overall) Proved 3 2 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 2 4 by DeepPolyR at [0, 8]=[-1.397530, 1.110187]
(DeepPolyR) Proved 3 2 4 by DeepPolyR at [0, 8]=[-1.397530, -1.397530]
(DeepPolyR) Proved 3 2 4 by DeepPolyR at [1, 8]=[0.065305, 1.110187]
(Overall) Proved 3 2 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 2 5 by DeepPolyR at [0, 8]=[-1.018760, 1.488957]
(DeepPolyR) Proved 3 2 5 by DeepPolyR at [0, 8]=[-1.018760, -0.078366]
(DeepPolyR) Proved 3 2 5 by DeepPolyR at [8, 8]=[1.488957, 1.488957]
(Overall) Proved 3 2 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 2 6 by DeepPolyR at [0, 8]=[-0.901211, 1.606506]
(DeepPolyR) Proved 3 2 6 by DeepPolyR at [0, 8]=[-0.901211, 0.352648]
(DeepPolyR) Proved 3 2 6 by DeepPolyR at [2, 8]=[0.561624, 1.606506]
(Overall) Proved 3 2 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 2 7 by DeepPolyR at [0, 8]=[-1.057943, 1.449774]
(DeepPolyR) Proved 3 2 7 by DeepPolyR at [0, 8]=[-1.057943, -0.013061]
(DeepPolyR) Proved 3 2 7 by DeepPolyR at [8, 8]=[1.449774, 1.449774]
(Overall) Proved 3 2 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 2 8 by DeepPolyR at [0, 8]=[-1.110187, 1.397530]
(DeepPolyR) Proved 3 2 8 by DeepPolyR at [0, 8]=[-1.110187, 0.143671]
(DeepPolyR) Proved 3 2 8 by DeepPolyR at [2, 8]=[0.457136, 1.397530]
(Overall) Proved 3 2 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 2 9 by DeepPolyR at [0, 8]=[-1.593445, 0.182854]
(DeepPolyR) Proved 3 2 9 by DeepPolyR at [0, 8]=[-1.593445, -1.071004]
(DeepPolyR) Proved 3 2 9 by DeepPolyR at [7, 8]=[-0.653051, 0.182854]
(Overall) Proved 3 2 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 3 3 0 by DeepPolyR at [0, 8]=[-1.279980, 1.227736]
(DeepPolyR) Proved 3 3 0 by DeepPolyR at [0, 8]=[-1.279980, -1.279980]
(DeepPolyR) Proved 3 3 0 by DeepPolyR at [1, 8]=[0.182854, 1.227736]
(Divide and Counter) Proved 3 3 0 by DeepPoly at [0]=-1.279980
(Divide and Counter) Proved 3 3 0 by DeepPolyR at [1, 8]=[0.182854, 1.227736]
(Overall) Proved 3 3 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 1 by DeepPolyR at [0, 8]=[-0.744478, 1.345286]
(DeepPolyR) Proved 3 3 1 by DeepPolyR at [0, 8]=[-0.744478, 0.091427]
(DeepPolyR) Proved 3 3 1 by DeepPolyR at [2, 8]=[0.875089, 1.345286]
(Overall) Proved 3 3 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 2 by DeepPolyR at [0, 8]=[-0.848967, 1.658750]
(DeepPolyR) Proved 3 3 2 by DeepPolyR at [0, 8]=[-0.848967, -0.013061]
(DeepPolyR) Proved 3 3 2 by DeepPolyR at [8, 8]=[1.658750, 1.658750]
(Overall) Proved 3 3 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 3 by DeepPolyR at [0, 8]=[-1.528140, 0.979577]
(DeepPolyR) Proved 3 3 3 by DeepPolyR at [0, 8]=[-1.528140, -1.528140]
(DeepPolyR) Proved 3 3 3 by DeepPolyR at [1, 8]=[0.039183, 0.979577]
(Overall) Proved 3 3 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 4 by DeepPolyR at [0, 8]=[-0.992638, 1.515079]
(DeepPolyR) Proved 3 3 4 by DeepPolyR at [0, 8]=[-0.992638, 0.261221]
(DeepPolyR) Proved 3 3 4 by DeepPolyR at [2, 8]=[0.470197, 1.515079]
(Overall) Proved 3 3 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 5 by DeepPolyR at [0, 8]=[-1.266919, 0.822845]
(DeepPolyR) Proved 3 3 5 by DeepPolyR at [0, 8]=[-1.266919, -0.848967]
(DeepPolyR) Proved 3 3 5 by DeepPolyR at [7, 8]=[-0.013061, 0.822845]
(Overall) Proved 3 3 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 6 by DeepPolyR at [0, 8]=[-1.410591, 0.679173]
(DeepPolyR) Proved 3 3 6 by DeepPolyR at [0, 8]=[-1.410591, -0.888150]
(DeepPolyR) Proved 3 3 6 by DeepPolyR at [7, 8]=[-0.156732, 0.679173]
(Overall) Proved 3 3 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 7 by DeepPolyR at [0, 8]=[-1.580384, 0.927333]
(DeepPolyR) Proved 3 3 7 by DeepPolyR at [0, 8]=[-1.580384, -1.580384]
(DeepPolyR) Proved 3 3 7 by DeepPolyR at [1, 8]=[0.039183, 0.927333]
(Overall) Proved 3 3 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 8 by DeepPolyR at [0, 8]=[-1.071004, 1.436713]
(DeepPolyR) Proved 3 3 8 by DeepPolyR at [0, 8]=[-1.071004, -0.026122]
(DeepPolyR) Proved 3 3 8 by DeepPolyR at [8, 8]=[1.436713, 1.436713]
(Overall) Proved 3 3 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 9 by DeepPolyR at [0, 8]=[-1.632628, 0.875089]
(DeepPolyR) Proved 3 3 9 by DeepPolyR at [0, 8]=[-1.632628, -1.632628]
(DeepPolyR) Proved 3 3 9 by DeepPolyR at [1, 8]=[0.013061, 0.875089]
(Overall) Proved 3 3 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 0 by DeepPolyR at [0, 8]=[-0.901211, 1.606506]
(DeepPolyR) Proved 3 4 0 by DeepPolyR at [0, 8]=[-0.901211, -0.013061]
(DeepPolyR) Proved 3 4 0 by DeepPolyR at [8, 8]=[1.606506, 1.606506]
(Overall) Proved 3 4 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 1 by DeepPolyR at [0, 8]=[-1.084065, 1.423652]
(DeepPolyR) Proved 3 4 1 by DeepPolyR at [0, 8]=[-1.084065, -0.039183]
(DeepPolyR) Proved 3 4 1 by DeepPolyR at [8, 8]=[1.423652, 1.423652]
(Overall) Proved 3 4 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 2 by DeepPolyR at [0, 8]=[-1.018760, 1.488957]
(DeepPolyR) Proved 3 4 2 by DeepPolyR at [0, 8]=[-1.018760, -0.078366]
(DeepPolyR) Proved 3 4 2 by DeepPolyR at [8, 8]=[1.488957, 1.488957]
(Overall) Proved 3 4 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 3 by DeepPolyR at [0, 8]=[-1.110187, 1.397530]
(DeepPolyR) Proved 3 4 3 by DeepPolyR at [0, 8]=[-1.110187, 0.143671]
(DeepPolyR) Proved 3 4 3 by DeepPolyR at [2, 8]=[0.457136, 1.397530]
(Overall) Proved 3 4 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 4 by DeepPolyR at [0, 8]=[-1.044882, 1.462835]
(DeepPolyR) Proved 3 4 4 by DeepPolyR at [0, 8]=[-1.044882, 0.208976]
(DeepPolyR) Proved 3 4 4 by DeepPolyR at [2, 8]=[0.417953, 1.462835]
(Overall) Proved 3 4 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 5 by DeepPolyR at [0, 8]=[-1.645689, 0.444075]
(DeepPolyR) Proved 3 4 5 by DeepPolyR at [0, 8]=[-1.645689, -1.018760]
(DeepPolyR) Proved 3 4 5 by DeepPolyR at [7, 8]=[-0.391831, 0.444075]
(Overall) Proved 3 4 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 6 by DeepPolyR at [0, 8]=[-1.645689, 0.235098]
(DeepPolyR) Proved 3 4 6 by DeepPolyR at [0, 8]=[-1.645689, -1.018760]
(DeepPolyR) Proved 3 4 6 by DeepPolyR at [7, 8]=[-0.600807, 0.235098]
(Overall) Proved 3 4 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 7 by DeepPolyR at [0, 8]=[-0.979577, 1.528140]
(DeepPolyR) Proved 3 4 7 by DeepPolyR at [0, 8]=[-0.979577, 0.274282]
(DeepPolyR) Proved 3 4 7 by DeepPolyR at [2, 8]=[0.483258, 1.528140]
(Overall) Proved 3 4 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 8 by DeepPolyR at [0, 8]=[-1.619567, 0.470197]
(DeepPolyR) Proved 3 4 8 by DeepPolyR at [0, 8]=[-1.619567, -0.992638]
(DeepPolyR) Proved 3 4 8 by DeepPolyR at [7, 8]=[-0.365709, 0.470197]
(Overall) Proved 3 4 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 9 by DeepPolyR at [0, 8]=[-1.227736, 1.279980]
(DeepPolyR) Proved 3 4 9 by DeepPolyR at [0, 8]=[-1.227736, 0.026122]
(DeepPolyR) Proved 3 4 9 by DeepPolyR at [2, 8]=[0.417953, 1.279980]
(Overall) Proved 3 4 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 0 by DeepPolyR at [0, 8]=[-0.992638, 1.515079]
(DeepPolyR) Proved 3 5 0 by DeepPolyR at [0, 8]=[-0.992638, 0.261221]
(DeepPolyR) Proved 3 5 0 by DeepPolyR at [2, 8]=[0.470197, 1.515079]
(Overall) Proved 3 5 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 1 by DeepPolyR at [0, 8]=[-1.306103, 1.201614]
(DeepPolyR) Proved 3 5 1 by DeepPolyR at [0, 8]=[-1.306103, -1.306103]
(DeepPolyR) Proved 3 5 1 by DeepPolyR at [1, 8]=[0.156732, 1.201614]
(Overall) Proved 3 5 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 2 by DeepPolyR at [0, 8]=[-1.358347, 0.731417]
(DeepPolyR) Proved 3 5 2 by DeepPolyR at [0, 8]=[-1.358347, -0.888150]
(DeepPolyR) Proved 3 5 2 by DeepPolyR at [7, 8]=[-0.104488, 0.731417]
(Overall) Proved 3 5 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 3 by DeepPolyR at [0, 8]=[-0.953455, 1.554262]
(DeepPolyR) Proved 3 5 3 by DeepPolyR at [0, 8]=[-0.953455, -0.013061]
(DeepPolyR) Proved 3 5 3 by DeepPolyR at [8, 8]=[1.554262, 1.554262]
(Overall) Proved 3 5 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 4 by DeepPolyR at [0, 8]=[-1.057943, 1.449774]
(DeepPolyR) Proved 3 5 4 by DeepPolyR at [0, 8]=[-1.057943, -1.057943]
(DeepPolyR) Proved 3 5 4 by DeepPolyR at [1, 8]=[0.195915, 1.449774]
(Overall) Proved 3 5 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 5 by DeepPolyR at [0, 8]=[-1.097126, 1.410591]
(DeepPolyR) Proved 3 5 5 by DeepPolyR at [0, 8]=[-1.097126, -0.052244]
(DeepPolyR) Proved 3 5 5 by DeepPolyR at [8, 8]=[1.410591, 1.410591]
(Overall) Proved 3 5 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 6 by DeepPolyR at [0, 8]=[-0.966516, 1.541201]
(DeepPolyR) Proved 3 5 6 by DeepPolyR at [0, 8]=[-0.966516, 0.287343]
(DeepPolyR) Proved 3 5 6 by DeepPolyR at [2, 8]=[0.496319, 1.541201]
(Overall) Proved 3 5 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 7 by DeepPolyR at [0, 8]=[-1.397530, 1.110187]
(DeepPolyR) Proved 3 5 7 by DeepPolyR at [0, 8]=[-1.397530, -1.397530]
(DeepPolyR) Proved 3 5 7 by DeepPolyR at [1, 8]=[0.065305, 1.110187]
(Overall) Proved 3 5 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 8 by DeepPolyR at [0, 8]=[-1.658750, 0.848967]
(DeepPolyR) Proved 3 5 8 by DeepPolyR at [0, 8]=[-1.658750, -1.658750]
(DeepPolyR) Proved 3 5 8 by DeepPolyR at [1, 8]=[0.000000, 0.848967]
(Overall) Proved 3 5 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 9 by DeepPolyR at [0, 8]=[-1.645689, 0.444075]
(DeepPolyR) Proved 3 5 9 by DeepPolyR at [0, 8]=[-1.645689, -1.018760]
(DeepPolyR) Proved 3 5 9 by DeepPolyR at [7, 8]=[-0.391831, 0.444075]
(Overall) Proved 3 5 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 0 by DeepPolyR at [0, 8]=[-0.888150, 1.619567]
(DeepPolyR) Proved 3 6 0 by DeepPolyR at [0, 8]=[-0.888150, -0.026122]
(DeepPolyR) Proved 3 6 0 by DeepPolyR at [8, 8]=[1.619567, 1.619567]
(Overall) Proved 3 6 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 1 by DeepPolyR at [0, 8]=[-1.632628, 0.875089]
(DeepPolyR) Proved 3 6 1 by DeepPolyR at [0, 8]=[-1.632628, -0.378770]
(DeepPolyR) Proved 3 6 1 by DeepPolyR at [8, 8]=[0.875089, 0.875089]
(Overall) Proved 3 6 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 2 by DeepPolyR at [0, 8]=[-1.175492, 1.332225]
(DeepPolyR) Proved 3 6 2 by DeepPolyR at [0, 8]=[-1.175492, -1.175492]
(DeepPolyR) Proved 3 6 2 by DeepPolyR at [1, 8]=[0.078366, 1.332225]
(Overall) Proved 3 6 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 3 by DeepPolyR at [0, 8]=[-0.404892, 1.475896]
(DeepPolyR) Proved 3 6 3 by DeepPolyR at [0, 8]=[-0.404892, 0.431014]
(DeepPolyR) Proved 3 6 3 by DeepPolyR at [2, 8]=[0.848967, 1.475896]
(Overall) Proved 3 6 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 4 by DeepPolyR at [0, 8]=[-0.966516, 1.541201]
(DeepPolyR) Proved 3 6 4 by DeepPolyR at [0, 8]=[-0.966516, 0.287343]
(DeepPolyR) Proved 3 6 4 by DeepPolyR at [2, 8]=[0.496319, 1.541201]
(Overall) Proved 3 6 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 5 by DeepPolyR at [0, 8]=[-1.580384, 0.927333]
(DeepPolyR) Proved 3 6 5 by DeepPolyR at [0, 8]=[-1.580384, -0.535502]
(DeepPolyR) Proved 3 6 5 by DeepPolyR at [7, 8]=[-0.326526, 0.927333]
(Overall) Proved 3 6 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 6 by DeepPolyR at [0, 8]=[-1.306103, 0.783662]
(DeepPolyR) Proved 3 6 6 by DeepPolyR at [0, 8]=[-1.306103, -0.862028]
(DeepPolyR) Proved 3 6 6 by DeepPolyR at [7, 8]=[-0.052244, 0.783662]
(Overall) Proved 3 6 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 7 by DeepPolyR at [0, 8]=[-1.044882, 1.462835]
(DeepPolyR) Proved 3 6 7 by DeepPolyR at [0, 8]=[-1.044882, 0.208976]
(DeepPolyR) Proved 3 6 7 by DeepPolyR at [2, 8]=[0.417953, 1.462835]
(Overall) Proved 3 6 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 8 by DeepPolyR at [0, 8]=[-1.358347, 0.731417]
(DeepPolyR) Proved 3 6 8 by DeepPolyR at [0, 8]=[-1.358347, -0.888150]
(DeepPolyR) Proved 3 6 8 by DeepPolyR at [7, 8]=[-0.104488, 0.731417]
(Overall) Proved 3 6 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 9 by DeepPolyR at [0, 8]=[-0.117549, 1.658750]
(DeepPolyR) Proved 3 6 9 by DeepPolyR at [0, 8]=[-0.117549, 0.718356]
(DeepPolyR) Proved 3 6 9 by DeepPolyR at [2, 8]=[1.136309, 1.658750]
(Overall) Proved 3 6 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 7 0 by DeepPolyR at [0, 8]=[-0.862028, 1.645689]
(DeepPolyR) Proved 3 7 0 by DeepPolyR at [0, 8]=[-0.862028, 0.391831]
(DeepPolyR) Proved 3 7 0 by DeepPolyR at [2, 8]=[0.600807, 1.645689]
(Overall) Proved 3 7 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 7 1 by DeepPolyR at [0, 8]=[-0.835906, 1.253858]
(DeepPolyR) Proved 3 7 1 by DeepPolyR at [0, 8]=[-0.835906, 0.000000]
(DeepPolyR) Proved 3 7 1 by DeepPolyR at [2, 8]=[0.835906, 1.253858]
(Overall) Proved 3 7 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 7 2 by DeepPolyR at [0, 8]=[-1.253858, 1.253858]
(DeepPolyR) Proved 3 7 2 by DeepPolyR at [0, 8]=[-1.253858, 0.000000]
(DeepPolyR) Proved 3 7 2 by DeepPolyR at [2, 8]=[0.417953, 1.253858]
(Overall) Proved 3 7 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 7 3 by DeepPolyR at [0, 8]=[-1.319164, 0.770600]
(DeepPolyR) Proved 3 7 3 by DeepPolyR at [0, 8]=[-1.319164, -0.848967]
(DeepPolyR) Proved 3 7 3 by DeepPolyR at [7, 8]=[-0.065305, 0.770600]
(Overall) Proved 3 7 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 7 4 by DeepPolyR at [0, 8]=[-1.645689, 0.862028]
(DeepPolyR) Proved 3 7 4 by DeepPolyR at [0, 8]=[-1.645689, -1.645689]
(DeepPolyR) Proved 3 7 4 by DeepPolyR at [1, 8]=[0.000000, 0.862028]
(Overall) Proved 3 7 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 7 5 by DeepPolyR at [0, 8]=[-0.339587, 1.541201]
(DeepPolyR) Proved 3 7 5 by DeepPolyR at [0, 8]=[-0.339587, 0.496319]
(DeepPolyR) Proved 3 7 5 by DeepPolyR at [2, 8]=[0.914272, 1.541201]
(Overall) Proved 3 7 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 7 6 by DeepPolyR at [0, 8]=[-0.718356, 1.371408]
(DeepPolyR) Proved 3 7 6 by DeepPolyR at [0, 8]=[-0.718356, 0.117549]
(DeepPolyR) Proved 3 7 6 by DeepPolyR at [2, 8]=[0.848967, 1.371408]
(Overall) Proved 3 7 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 7 7 by DeepPolyR at [0, 8]=[-1.005699, 1.502018]
(DeepPolyR) Proved 3 7 7 by DeepPolyR at [0, 8]=[-1.005699, -0.065305]
(DeepPolyR) Proved 3 7 7 by DeepPolyR at [8, 8]=[1.502018, 1.502018]
(Overall) Proved 3 7 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 7 8 by DeepPolyR at [0, 8]=[-0.587746, 1.502018]
(DeepPolyR) Proved 3 7 8 by DeepPolyR at [0, 8]=[-0.587746, 0.248159]
(DeepPolyR) Proved 3 7 8 by DeepPolyR at [2, 8]=[0.875089, 1.502018]
(Overall) Proved 3 7 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 7 9 by DeepPolyR at [0, 8]=[-1.658750, 0.222037]
(DeepPolyR) Proved 3 7 9 by DeepPolyR at [0, 8]=[-1.658750, -1.031821]
(DeepPolyR) Proved 3 7 9 by DeepPolyR at [7, 8]=[-0.613868, 0.222037]
(Overall) Proved 3 7 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 3 8 0 by DeepPolyR at [0, 8]=[-0.862028, 1.645689]
(DeepPolyR) Proved 3 8 0 by DeepPolyR at [0, 8]=[-0.862028, -0.013061]
(DeepPolyR) Fail to prove 3 8 0 by DeepPolyR at [8, 8]=[1.645689, 1.645689]
(Divide and Counter) Proved 3 8 0 by DeepPolyR at [0, 7]=[-0.862028, -0.013061]
(Divide and Counter) Fail to prove 3 8 0 by DeepPoly at [8]=1.645689
(Overall) Fail to prove 3 8 0 with all masks. Summary: 0 1 0 

(Divide and Counter) Fail to prove 3 8 1 by DeepPolyR at [0, 8]=[-1.580384, 0.509380]
(DeepPolyR) Proved 3 8 1 by DeepPolyR at [0, 8]=[-1.580384, -0.953455]
(DeepPolyR) Fail to prove 3 8 1 by DeepPolyR at [7, 8]=[-0.326526, 0.509380]
(Divide and Counter) Proved 3 8 1 by DeepPolyR at [0, 6]=[-1.580384, -0.953455]
(Divide and Counter) Fail to prove 3 8 1 by DeepPolyR at [7, 8]=[-0.326526, 0.509380]
(Divide and Counter) Fail to prove 3 8 1 by DeepPoly at [7]=-0.326526
(Overall) Fail to prove 3 8 1 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 3 8 2 by DeepPolyR at [0, 8]=[-0.509380, 1.580384]
(DeepPolyR) Proved 3 8 2 by DeepPolyR at [0, 8]=[-0.509380, 0.326526]
(DeepPolyR) Proved 3 8 2 by DeepPolyR at [2, 8]=[0.953455, 1.580384]
(Overall) Proved 3 8 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 8 3 by DeepPolyR at [0, 8]=[-1.175492, 1.332225]
(DeepPolyR) Proved 3 8 3 by DeepPolyR at [0, 8]=[-1.175492, -1.175492]
(DeepPolyR) Proved 3 8 3 by DeepPolyR at [1, 8]=[0.078366, 1.332225]
(Overall) Proved 3 8 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 8 4 by DeepPolyR at [0, 8]=[-0.992638, 1.515079]
(DeepPolyR) Proved 3 8 4 by DeepPolyR at [0, 8]=[-0.992638, -0.052244]
(DeepPolyR) Proved 3 8 4 by DeepPolyR at [8, 8]=[1.515079, 1.515079]
(Overall) Proved 3 8 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 8 5 by DeepPolyR at [0, 8]=[-0.692234, 1.397530]
(DeepPolyR) Proved 3 8 5 by DeepPolyR at [0, 8]=[-0.692234, 0.143671]
(DeepPolyR) Proved 3 8 5 by DeepPolyR at [2, 8]=[0.875089, 1.397530]
(Overall) Proved 3 8 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 8 6 by DeepPolyR at [0, 8]=[-0.496319, 1.593445]
(DeepPolyR) Proved 3 8 6 by DeepPolyR at [0, 8]=[-0.496319, 0.339587]
(DeepPolyR) Proved 3 8 6 by DeepPolyR at [2, 8]=[0.966516, 1.593445]
(Overall) Proved 3 8 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 8 7 by DeepPolyR at [0, 8]=[-1.671811, 0.835906]
(DeepPolyR) Proved 3 8 7 by DeepPolyR at [0, 8]=[-1.671811, -1.671811]
(DeepPolyR) Proved 3 8 7 by DeepPolyR at [1, 8]=[0.000000, 0.835906]
(Overall) Proved 3 8 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 3 8 8 by DeepPolyR at [0, 8]=[-0.901211, 1.606506]
(DeepPolyR) Proved 3 8 8 by DeepPolyR at [0, 8]=[-0.901211, -0.013061]
(DeepPolyR) Fail to prove 3 8 8 by DeepPolyR at [8, 8]=[1.606506, 1.606506]
(Divide and Counter) Proved 3 8 8 by DeepPolyR at [0, 7]=[-0.901211, -0.013061]
(Divide and Counter) Fail to prove 3 8 8 by DeepPoly at [8]=1.606506
(Overall) Fail to prove 3 8 8 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 3 8 9 by DeepPolyR at [0, 8]=[-0.378770, 1.502018]
(DeepPolyR) Proved 3 8 9 by DeepPolyR at [0, 8]=[-0.378770, 0.457136]
(DeepPolyR) Proved 3 8 9 by DeepPolyR at [2, 8]=[0.875089, 1.502018]
(Overall) Proved 3 8 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 3 9 0 by DeepPolyR at [0, 8]=[-1.345286, 1.162431]
(DeepPolyR) Proved 3 9 0 by DeepPolyR at [0, 8]=[-1.345286, -1.345286]
(DeepPolyR) Proved 3 9 0 by DeepPolyR at [1, 8]=[0.117549, 1.162431]
(Divide and Counter) Proved 3 9 0 by DeepPoly at [0]=-1.345286
(Divide and Counter) Proved 3 9 0 by DeepPolyR at [1, 8]=[0.117549, 1.162431]
(Overall) Proved 3 9 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 3 9 1 by DeepPolyR at [0, 8]=[-0.666112, 1.423652]
(DeepPolyR) Proved 3 9 1 by DeepPolyR at [0, 8]=[-0.666112, 0.169793]
(DeepPolyR) Proved 3 9 1 by DeepPolyR at [2, 8]=[0.901211, 1.423652]
(Divide and Counter) Proved 3 9 1 by DeepPolyR at [0, 1]=[-0.666112, 0.169793]
(Divide and Counter) Proved 3 9 1 by DeepPolyR at [2, 8]=[0.901211, 1.423652]
(Overall) Proved 3 9 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 2 by DeepPolyR at [0, 8]=[-1.462835, 0.626929]
(DeepPolyR) Proved 3 9 2 by DeepPolyR at [0, 8]=[-1.462835, -0.940394]
(DeepPolyR) Proved 3 9 2 by DeepPolyR at [7, 8]=[-0.208976, 0.626929]
(Overall) Proved 3 9 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 3 by DeepPolyR at [0, 8]=[-1.227736, 1.279980]
(DeepPolyR) Proved 3 9 3 by DeepPolyR at [0, 8]=[-1.227736, -0.182854]
(DeepPolyR) Proved 3 9 3 by DeepPolyR at [8, 8]=[1.279980, 1.279980]
(Overall) Proved 3 9 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 4 by DeepPolyR at [0, 8]=[-1.071004, 1.436713]
(DeepPolyR) Proved 3 9 4 by DeepPolyR at [0, 8]=[-1.071004, -1.071004]
(DeepPolyR) Proved 3 9 4 by DeepPolyR at [1, 8]=[0.182854, 1.436713]
(Overall) Proved 3 9 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 5 by DeepPolyR at [0, 8]=[-1.606506, 0.901211]
(DeepPolyR) Proved 3 9 5 by DeepPolyR at [0, 8]=[-1.606506, -0.561624]
(DeepPolyR) Proved 3 9 5 by DeepPolyR at [7, 8]=[-0.352648, 0.901211]
(Overall) Proved 3 9 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 6 by DeepPolyR at [0, 8]=[-1.554262, 0.953455]
(DeepPolyR) Proved 3 9 6 by DeepPolyR at [0, 8]=[-1.554262, -0.300404]
(DeepPolyR) Proved 3 9 6 by DeepPolyR at [8, 8]=[0.953455, 0.953455]
(Overall) Proved 3 9 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 7 by DeepPolyR at [0, 8]=[-1.371408, 1.136309]
(DeepPolyR) Proved 3 9 7 by DeepPolyR at [0, 8]=[-1.371408, -1.371408]
(DeepPolyR) Proved 3 9 7 by DeepPolyR at [1, 8]=[0.091427, 1.136309]
(Overall) Proved 3 9 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 8 by DeepPolyR at [0, 8]=[-1.671811, 0.835906]
(DeepPolyR) Proved 3 9 8 by DeepPolyR at [0, 8]=[-1.671811, -1.671811]
(DeepPolyR) Proved 3 9 8 by DeepPolyR at [1, 8]=[0.000000, 0.835906]
(Overall) Proved 3 9 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 9 by DeepPolyR at [0, 8]=[-1.358347, 1.149370]
(DeepPolyR) Proved 3 9 9 by DeepPolyR at [0, 8]=[-1.358347, -0.470197]
(DeepPolyR) Proved 3 9 9 by DeepPolyR at [7, 8]=[-0.104488, 1.149370]
(Overall) Proved 3 9 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 0 by DeepPolyR at [0, 8]=[-1.378526, 1.595373]
(DeepPolyR) Proved 4 0 0 by DeepPolyR at [0, 8]=[-1.378526, 0.108423]
(DeepPolyR) Proved 4 0 0 by DeepPolyR at [2, 8]=[0.542117, 1.595373]
(Overall) Proved 4 0 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 1 by DeepPolyR at [0, 8]=[-1.301080, 1.672818]
(DeepPolyR) Proved 4 0 1 by DeepPolyR at [0, 8]=[-1.301080, 0.185869]
(DeepPolyR) Proved 4 0 1 by DeepPolyR at [2, 8]=[0.557606, 1.672818]
(Overall) Proved 4 0 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 2 by DeepPolyR at [0, 8]=[-1.796730, 0.681518]
(DeepPolyR) Proved 4 0 2 by DeepPolyR at [0, 8]=[-1.796730, -1.053256]
(DeepPolyR) Proved 4 0 2 by DeepPolyR at [7, 8]=[-0.309781, 0.681518]
(Overall) Proved 4 0 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 3 by DeepPolyR at [0, 8]=[-1.657329, 1.316570]
(DeepPolyR) Proved 4 0 3 by DeepPolyR at [0, 8]=[-1.657329, -0.170380]
(DeepPolyR) Proved 4 0 3 by DeepPolyR at [8, 8]=[1.316570, 1.316570]
(Overall) Proved 4 0 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 4 by DeepPolyR at [0, 8]=[-1.750263, 0.480161]
(DeepPolyR) Proved 4 0 4 by DeepPolyR at [0, 8]=[-1.750263, -1.006788]
(DeepPolyR) Proved 4 0 4 by DeepPolyR at [7, 8]=[-0.511139, 0.480161]
(Overall) Proved 4 0 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 5 by DeepPolyR at [0, 8]=[-1.564394, 0.913854]
(DeepPolyR) Proved 4 0 5 by DeepPolyR at [0, 8]=[-1.564394, -1.006788]
(DeepPolyR) Proved 4 0 5 by DeepPolyR at [7, 8]=[-0.077445, 0.913854]
(Overall) Proved 4 0 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 6 by DeepPolyR at [0, 8]=[-1.208146, 1.765752]
(DeepPolyR) Proved 4 0 6 by DeepPolyR at [0, 8]=[-1.208146, -0.092934]
(DeepPolyR) Proved 4 0 6 by DeepPolyR at [8, 8]=[1.765752, 1.765752]
(Overall) Proved 4 0 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 7 by DeepPolyR at [0, 8]=[-1.967110, 1.006788]
(DeepPolyR) Proved 4 0 7 by DeepPolyR at [0, 8]=[-1.967110, -1.967110]
(DeepPolyR) Proved 4 0 7 by DeepPolyR at [1, 8]=[0.000000, 1.006788]
(Overall) Proved 4 0 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 8 by DeepPolyR at [0, 8]=[-1.424993, 1.548905]
(DeepPolyR) Proved 4 0 8 by DeepPolyR at [0, 8]=[-1.424993, 0.061956]
(DeepPolyR) Proved 4 0 8 by DeepPolyR at [2, 8]=[0.495650, 1.548905]
(Overall) Proved 4 0 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 9 by DeepPolyR at [0, 8]=[-1.208146, 1.765752]
(DeepPolyR) Proved 4 0 9 by DeepPolyR at [0, 8]=[-1.208146, -0.092934]
(DeepPolyR) Proved 4 0 9 by DeepPolyR at [8, 8]=[1.765752, 1.765752]
(Overall) Proved 4 0 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 1 0 by DeepPolyR at [0, 8]=[-1.936132, 1.037767]
(DeepPolyR) Proved 4 1 0 by DeepPolyR at [0, 8]=[-1.936132, -0.449183]
(DeepPolyR) Proved 4 1 0 by DeepPolyR at [8, 8]=[1.037767, 1.037767]
(Overall) Proved 4 1 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 1 1 by DeepPolyR at [0, 8]=[-1.734774, 0.743475]
(DeepPolyR) Proved 4 1 1 by DeepPolyR at [0, 8]=[-1.734774, -1.115212]
(DeepPolyR) Proved 4 1 1 by DeepPolyR at [7, 8]=[-0.247825, 0.743475]
(Overall) Proved 4 1 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 1 2 by DeepPolyR at [0, 8]=[-1.982599, 0.015489]
(DeepPolyR) Proved 4 1 2 by DeepPolyR at [0, 8]=[-1.982599, -1.471460]
(DeepPolyR) Proved 4 1 2 by DeepPolyR at [7, 8]=[-0.975810, 0.015489]
(Overall) Proved 4 1 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 1 3 by DeepPolyR at [0, 8]=[-1.610862, 1.363037]
(DeepPolyR) Proved 4 1 3 by DeepPolyR at [0, 8]=[-1.610862, -1.610862]
(DeepPolyR) Proved 4 1 3 by DeepPolyR at [1, 8]=[0.123912, 1.363037]
(Overall) Proved 4 1 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 1 4 by DeepPolyR at [0, 8]=[-1.750263, 1.223635]
(DeepPolyR) Proved 4 1 4 by DeepPolyR at [0, 8]=[-1.750263, -1.750263]
(DeepPolyR) Proved 4 1 4 by DeepPolyR at [1, 8]=[0.108423, 1.223635]
(Overall) Proved 4 1 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 1 5 by DeepPolyR at [0, 8]=[-1.332059, 1.641840]
(DeepPolyR) Proved 4 1 5 by DeepPolyR at [0, 8]=[-1.332059, 0.154891]
(DeepPolyR) Proved 4 1 5 by DeepPolyR at [2, 8]=[0.526628, 1.641840]
(Overall) Proved 4 1 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 1 6 by DeepPolyR at [0, 8]=[-1.208146, 1.765752]
(DeepPolyR) Proved 4 1 6 by DeepPolyR at [0, 8]=[-1.208146, 0.278803]
(DeepPolyR) Proved 4 1 6 by DeepPolyR at [2, 8]=[0.526628, 1.765752]
(Overall) Proved 4 1 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 1 7 by DeepPolyR at [0, 8]=[-1.874175, 0.232336]
(DeepPolyR) Proved 4 1 7 by DeepPolyR at [0, 8]=[-1.874175, -1.254613]
(DeepPolyR) Proved 4 1 7 by DeepPolyR at [7, 8]=[-0.758964, 0.232336]
(Overall) Proved 4 1 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 1 8 by DeepPolyR at [0, 8]=[-1.765752, 1.208146]
(DeepPolyR) Proved 4 1 8 by DeepPolyR at [0, 8]=[-1.765752, -0.278803]
(DeepPolyR) Proved 4 1 8 by DeepPolyR at [8, 8]=[1.208146, 1.208146]
(Overall) Proved 4 1 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 1 9 by DeepPolyR at [0, 8]=[-1.548905, 1.424993]
(DeepPolyR) Proved 4 1 9 by DeepPolyR at [0, 8]=[-1.548905, -1.548905]
(DeepPolyR) Proved 4 1 9 by DeepPolyR at [1, 8]=[0.185869, 1.424993]
(Overall) Proved 4 1 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 0 by DeepPolyR at [0, 8]=[-1.750263, 1.223635]
(DeepPolyR) Proved 4 2 0 by DeepPolyR at [0, 8]=[-1.750263, -0.511139]
(DeepPolyR) Proved 4 2 0 by DeepPolyR at [7, 8]=[-0.263314, 1.223635]
(Overall) Proved 4 2 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 1 by DeepPolyR at [0, 8]=[-1.610862, 1.363037]
(DeepPolyR) Proved 4 2 1 by DeepPolyR at [0, 8]=[-1.610862, -1.610862]
(DeepPolyR) Proved 4 2 1 by DeepPolyR at [1, 8]=[0.123912, 1.363037]
(Overall) Proved 4 2 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 2 by DeepPolyR at [0, 8]=[-1.099723, 1.874175]
(DeepPolyR) Proved 4 2 2 by DeepPolyR at [0, 8]=[-1.099723, 0.387226]
(DeepPolyR) Proved 4 2 2 by DeepPolyR at [2, 8]=[0.635051, 1.874175]
(Overall) Proved 4 2 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 3 by DeepPolyR at [0, 8]=[-1.130701, 1.843197]
(DeepPolyR) Proved 4 2 3 by DeepPolyR at [0, 8]=[-1.130701, -0.015489]
(DeepPolyR) Proved 4 2 3 by DeepPolyR at [8, 8]=[1.843197, 1.843197]
(Overall) Proved 4 2 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 4 by DeepPolyR at [0, 8]=[-1.455971, 1.517927]
(DeepPolyR) Proved 4 2 4 by DeepPolyR at [0, 8]=[-1.455971, 0.030978]
(DeepPolyR) Proved 4 2 4 by DeepPolyR at [2, 8]=[0.495650, 1.517927]
(Overall) Proved 4 2 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 5 by DeepPolyR at [0, 8]=[-1.548905, 1.424993]
(DeepPolyR) Proved 4 2 5 by DeepPolyR at [0, 8]=[-1.548905, -0.526628]
(DeepPolyR) Proved 4 2 5 by DeepPolyR at [7, 8]=[-0.061956, 1.424993]
(Overall) Proved 4 2 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 6 by DeepPolyR at [0, 8]=[-1.161679, 1.812219]
(DeepPolyR) Proved 4 2 6 by DeepPolyR at [0, 8]=[-1.161679, 0.325270]
(DeepPolyR) Proved 4 2 6 by DeepPolyR at [2, 8]=[0.573095, 1.812219]
(Overall) Proved 4 2 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 7 by DeepPolyR at [0, 8]=[-1.548905, 1.424993]
(DeepPolyR) Proved 4 2 7 by DeepPolyR at [0, 8]=[-1.548905, -0.526628]
(DeepPolyR) Proved 4 2 7 by DeepPolyR at [7, 8]=[-0.061956, 1.424993]
(Overall) Proved 4 2 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 8 by DeepPolyR at [0, 8]=[-1.967110, 1.006788]
(DeepPolyR) Proved 4 2 8 by DeepPolyR at [0, 8]=[-1.967110, -1.967110]
(DeepPolyR) Proved 4 2 8 by DeepPolyR at [1, 8]=[0.000000, 1.006788]
(Overall) Proved 4 2 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 9 by DeepPolyR at [0, 8]=[-1.146190, 1.827708]
(DeepPolyR) Proved 4 2 9 by DeepPolyR at [0, 8]=[-1.146190, -0.030978]
(DeepPolyR) Proved 4 2 9 by DeepPolyR at [8, 8]=[1.827708, 1.827708]
(Overall) Proved 4 2 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 0 by DeepPolyR at [0, 8]=[-1.827708, 1.146190]
(DeepPolyR) Proved 4 3 0 by DeepPolyR at [0, 8]=[-1.827708, -1.827708]
(DeepPolyR) Proved 4 3 0 by DeepPolyR at [1, 8]=[0.030978, 1.146190]
(Overall) Proved 4 3 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 1 by DeepPolyR at [0, 8]=[-1.378526, 1.595373]
(DeepPolyR) Proved 4 3 1 by DeepPolyR at [0, 8]=[-1.378526, 0.108423]
(DeepPolyR) Proved 4 3 1 by DeepPolyR at [2, 8]=[0.542117, 1.595373]
(Overall) Proved 4 3 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 2 by DeepPolyR at [0, 8]=[-1.703796, 1.270102]
(DeepPolyR) Proved 4 3 2 by DeepPolyR at [0, 8]=[-1.703796, -1.703796]
(DeepPolyR) Proved 4 3 2 by DeepPolyR at [1, 8]=[0.030978, 1.270102]
(Overall) Proved 4 3 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 3 by DeepPolyR at [0, 8]=[-1.270102, 1.703796]
(DeepPolyR) Proved 4 3 3 by DeepPolyR at [0, 8]=[-1.270102, -0.030978]
(DeepPolyR) Proved 4 3 3 by DeepPolyR at [8, 8]=[1.703796, 1.703796]
(Overall) Proved 4 3 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 4 by DeepPolyR at [0, 8]=[-1.672818, 1.301080]
(DeepPolyR) Proved 4 3 4 by DeepPolyR at [0, 8]=[-1.672818, -1.672818]
(DeepPolyR) Proved 4 3 4 by DeepPolyR at [1, 8]=[0.061956, 1.301080]
(Overall) Proved 4 3 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 5 by DeepPolyR at [0, 8]=[-1.517927, 1.455971]
(DeepPolyR) Proved 4 3 5 by DeepPolyR at [0, 8]=[-1.517927, -0.511139]
(DeepPolyR) Proved 4 3 5 by DeepPolyR at [7, 8]=[-0.030978, 1.455971]
(Overall) Proved 4 3 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 6 by DeepPolyR at [0, 8]=[-1.672818, 1.301080]
(DeepPolyR) Proved 4 3 6 by DeepPolyR at [0, 8]=[-1.672818, -1.672818]
(DeepPolyR) Proved 4 3 6 by DeepPolyR at [1, 8]=[0.061956, 1.301080]
(Overall) Proved 4 3 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 7 by DeepPolyR at [0, 8]=[-1.874175, 1.099723]
(DeepPolyR) Proved 4 3 7 by DeepPolyR at [0, 8]=[-1.874175, -0.387226]
(DeepPolyR) Proved 4 3 7 by DeepPolyR at [8, 8]=[1.099723, 1.099723]
(Overall) Proved 4 3 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 8 by DeepPolyR at [0, 8]=[-1.657329, 1.316570]
(DeepPolyR) Proved 4 3 8 by DeepPolyR at [0, 8]=[-1.657329, -1.657329]
(DeepPolyR) Proved 4 3 8 by DeepPolyR at [1, 8]=[0.077445, 1.316570]
(Overall) Proved 4 3 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 9 by DeepPolyR at [0, 8]=[-1.641840, 1.332059]
(DeepPolyR) Proved 4 3 9 by DeepPolyR at [0, 8]=[-1.641840, -0.154891]
(DeepPolyR) Proved 4 3 9 by DeepPolyR at [8, 8]=[1.332059, 1.332059]
(Overall) Proved 4 3 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 4 0 by DeepPolyR at [0, 8]=[-1.889665, 1.084234]
(DeepPolyR) Fail to prove 4 4 0 by DeepPolyR at [0, 8]=[-1.889665, -1.889665]
(DeepPolyR) Proved 4 4 0 by DeepPolyR at [1, 8]=[0.030978, 1.084234]
(Divide and Counter) Fail to prove 4 4 0 by DeepPoly at [0]=-1.889665
(Overall) Fail to prove 4 4 0 with all masks. Summary: 0 0 1 

(Divide and Counter) Proved 4 4 1 by DeepPolyR at [0, 8]=[-1.982599, 0.991299]
(DeepPolyR) Proved 4 4 1 by DeepPolyR at [0, 8]=[-1.982599, -0.495650]
(DeepPolyR) Proved 4 4 1 by DeepPolyR at [8, 8]=[0.991299, 0.991299]
(Overall) Proved 4 4 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 4 2 by DeepPolyR at [0, 8]=[-1.254613, 1.719285]
(DeepPolyR) Fail to prove 4 4 2 by DeepPolyR at [0, 8]=[-1.254613, 0.232336]
(DeepPolyR) Proved 4 4 2 by DeepPolyR at [2, 8]=[0.604073, 1.719285]
(Divide and Counter) Fail to prove 4 4 2 by DeepPolyR at [0, 1]=[-1.254613, 0.232336]
(Divide and Counter) Fail to prove 4 4 2 by DeepPoly at [0]=-1.254613
(Overall) Fail to prove 4 4 2 with all masks. Summary: 0 0 1 

(Divide and Counter) Fail to prove 4 4 3 by DeepPolyR at [0, 8]=[-1.750263, 1.223635]
(DeepPolyR) Fail to prove 4 4 3 by DeepPolyR at [0, 8]=[-1.750263, -1.750263]
(DeepPolyR) Proved 4 4 3 by DeepPolyR at [1, 8]=[0.108423, 1.223635]
(Divide and Counter) Fail to prove 4 4 3 by DeepPoly at [0]=-1.750263
(Overall) Fail to prove 4 4 3 with all masks. Summary: 0 0 1 

(Divide and Counter) Proved 4 4 4 by DeepPolyR at [0, 8]=[-1.579883, 1.394015]
(DeepPolyR) Proved 4 4 4 by DeepPolyR at [0, 8]=[-1.579883, -0.092934]
(DeepPolyR) Proved 4 4 4 by DeepPolyR at [8, 8]=[1.394015, 1.394015]
(Overall) Proved 4 4 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 4 5 by DeepPolyR at [0, 8]=[-1.734774, 1.239124]
(DeepPolyR) Fail to prove 4 4 5 by DeepPolyR at [0, 8]=[-1.734774, -1.734774]
(DeepPolyR) Proved 4 4 5 by DeepPolyR at [1, 8]=[0.000000, 1.239124]
(Divide and Counter) Fail to prove 4 4 5 by DeepPoly at [0]=-1.734774
(Overall) Fail to prove 4 4 5 with all masks. Summary: 0 0 1 

(Divide and Counter) Proved 4 4 6 by DeepPolyR at [0, 8]=[-1.781241, 0.697007]
(DeepPolyR) Proved 4 4 6 by DeepPolyR at [0, 8]=[-1.781241, -1.037767]
(DeepPolyR) Proved 4 4 6 by DeepPolyR at [7, 8]=[-0.294292, 0.697007]
(Overall) Proved 4 4 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 4 7 by DeepPolyR at [0, 8]=[-1.254613, 1.719285]
(DeepPolyR) Fail to prove 4 4 7 by DeepPolyR at [0, 8]=[-1.254613, 0.232336]
(DeepPolyR) Proved 4 4 7 by DeepPolyR at [2, 8]=[0.604073, 1.719285]
(Divide and Counter) Fail to prove 4 4 7 by DeepPolyR at [0, 1]=[-1.254613, 0.232336]
(Divide and Counter) Fail to prove 4 4 7 by DeepPoly at [0]=-1.254613
(Overall) Fail to prove 4 4 7 with all masks. Summary: 0 0 1 

(Divide and Counter) Proved 4 4 8 by DeepPolyR at [0, 8]=[-1.951621, 0.092934]
(DeepPolyR) Proved 4 4 8 by DeepPolyR at [0, 8]=[-1.951621, -1.394015]
(DeepPolyR) Proved 4 4 8 by DeepPolyR at [7, 8]=[-0.898365, 0.092934]
(Overall) Proved 4 4 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 4 9 by DeepPolyR at [0, 8]=[-1.239124, 1.734774]
(DeepPolyR) Fail to prove 4 4 9 by DeepPolyR at [0, 8]=[-1.239124, -1.239124]
(DeepPolyR) Proved 4 4 9 by DeepPolyR at [1, 8]=[0.247825, 1.734774]
(Divide and Counter) Fail to prove 4 4 9 by DeepPoly at [0]=-1.239124
(Overall) Fail to prove 4 4 9 with all masks. Summary: 0 0 1 

(Divide and Counter) Proved 4 5 0 by DeepPolyR at [0, 8]=[-1.455971, 1.517927]
(DeepPolyR) Proved 4 5 0 by DeepPolyR at [0, 8]=[-1.455971, 0.030978]
(DeepPolyR) Proved 4 5 0 by DeepPolyR at [2, 8]=[0.495650, 1.517927]
(Overall) Proved 4 5 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 1 by DeepPolyR at [0, 8]=[-1.657329, 1.316570]
(DeepPolyR) Proved 4 5 1 by DeepPolyR at [0, 8]=[-1.657329, -1.657329]
(DeepPolyR) Proved 4 5 1 by DeepPolyR at [1, 8]=[0.077445, 1.316570]
(Overall) Proved 4 5 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 2 by DeepPolyR at [0, 8]=[-1.703796, 1.270102]
(DeepPolyR) Proved 4 5 2 by DeepPolyR at [0, 8]=[-1.703796, -0.216847]
(DeepPolyR) Proved 4 5 2 by DeepPolyR at [8, 8]=[1.270102, 1.270102]
(Overall) Proved 4 5 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 3 by DeepPolyR at [0, 8]=[-1.223635, 1.750263]
(DeepPolyR) Proved 4 5 3 by DeepPolyR at [0, 8]=[-1.223635, -0.108423]
(DeepPolyR) Proved 4 5 3 by DeepPolyR at [8, 8]=[1.750263, 1.750263]
(Overall) Proved 4 5 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 4 by DeepPolyR at [0, 8]=[-1.967110, 1.006788]
(DeepPolyR) Proved 4 5 4 by DeepPolyR at [0, 8]=[-1.967110, -1.967110]
(DeepPolyR) Proved 4 5 4 by DeepPolyR at [1, 8]=[0.000000, 1.006788]
(Overall) Proved 4 5 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 5 by DeepPolyR at [0, 8]=[-1.424993, 1.548905]
(DeepPolyR) Proved 4 5 5 by DeepPolyR at [0, 8]=[-1.424993, -0.185869]
(DeepPolyR) Proved 4 5 5 by DeepPolyR at [8, 8]=[1.548905, 1.548905]
(Overall) Proved 4 5 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 6 by DeepPolyR at [0, 8]=[-1.719285, 1.254613]
(DeepPolyR) Proved 4 5 6 by DeepPolyR at [0, 8]=[-1.719285, -1.719285]
(DeepPolyR) Proved 4 5 6 by DeepPolyR at [1, 8]=[0.015489, 1.254613]
(Overall) Proved 4 5 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 7 by DeepPolyR at [0, 8]=[-1.781241, 1.192657]
(DeepPolyR) Proved 4 5 7 by DeepPolyR at [0, 8]=[-1.781241, -1.781241]
(DeepPolyR) Proved 4 5 7 by DeepPolyR at [1, 8]=[0.077445, 1.192657]
(Overall) Proved 4 5 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 8 by DeepPolyR at [0, 8]=[-1.223635, 1.750263]
(DeepPolyR) Proved 4 5 8 by DeepPolyR at [0, 8]=[-1.223635, -0.108423]
(DeepPolyR) Proved 4 5 8 by DeepPolyR at [8, 8]=[1.750263, 1.750263]
(Overall) Proved 4 5 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 9 by DeepPolyR at [0, 8]=[-1.502438, 1.471460]
(DeepPolyR) Proved 4 5 9 by DeepPolyR at [0, 8]=[-1.502438, -0.015489]
(DeepPolyR) Proved 4 5 9 by DeepPolyR at [8, 8]=[1.471460, 1.471460]
(Overall) Proved 4 5 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 6 0 by DeepPolyR at [0, 8]=[-1.440482, 1.533416]
(DeepPolyR) Proved 4 6 0 by DeepPolyR at [0, 8]=[-1.440482, -0.201358]
(DeepPolyR) Fail to prove 4 6 0 by DeepPolyR at [8, 8]=[1.533416, 1.533416]
(Divide and Counter) Proved 4 6 0 by DeepPolyR at [0, 7]=[-1.440482, -0.201358]
(Divide and Counter) Fail to prove 4 6 0 by DeepPoly at [8]=1.533416
(Overall) Fail to prove 4 6 0 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 4 6 1 by DeepPolyR at [0, 8]=[-1.502438, 0.975810]
(DeepPolyR) Proved 4 6 1 by DeepPolyR at [0, 8]=[-1.502438, -1.006788]
(DeepPolyR) Proved 4 6 1 by DeepPolyR at [7, 8]=[-0.015489, 0.975810]
(Overall) Proved 4 6 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 2 by DeepPolyR at [0, 8]=[-1.579883, 1.394015]
(DeepPolyR) Proved 4 6 2 by DeepPolyR at [0, 8]=[-1.579883, -1.579883]
(DeepPolyR) Proved 4 6 2 by DeepPolyR at [1, 8]=[0.154891, 1.394015]
(Overall) Proved 4 6 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 3 by DeepPolyR at [0, 8]=[-1.270102, 1.703796]
(DeepPolyR) Proved 4 6 3 by DeepPolyR at [0, 8]=[-1.270102, -1.270102]
(DeepPolyR) Proved 4 6 3 by DeepPolyR at [1, 8]=[0.216847, 1.703796]
(Overall) Proved 4 6 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 4 by DeepPolyR at [0, 8]=[-1.579883, 1.394015]
(DeepPolyR) Proved 4 6 4 by DeepPolyR at [0, 8]=[-1.579883, -1.579883]
(DeepPolyR) Proved 4 6 4 by DeepPolyR at [1, 8]=[0.154891, 1.394015]
(Overall) Proved 4 6 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 5 by DeepPolyR at [0, 8]=[-1.827708, 1.146190]
(DeepPolyR) Proved 4 6 5 by DeepPolyR at [0, 8]=[-1.827708, -1.827708]
(DeepPolyR) Proved 4 6 5 by DeepPolyR at [1, 8]=[0.030978, 1.146190]
(Overall) Proved 4 6 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 6 by DeepPolyR at [0, 8]=[-1.347548, 1.626351]
(DeepPolyR) Proved 4 6 6 by DeepPolyR at [0, 8]=[-1.347548, -0.108423]
(DeepPolyR) Proved 4 6 6 by DeepPolyR at [8, 8]=[1.626351, 1.626351]
(Overall) Proved 4 6 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 7 by DeepPolyR at [0, 8]=[-1.982599, 0.991299]
(DeepPolyR) Proved 4 6 7 by DeepPolyR at [0, 8]=[-1.982599, -1.982599]
(DeepPolyR) Proved 4 6 7 by DeepPolyR at [1, 8]=[0.000000, 0.991299]
(Overall) Proved 4 6 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 8 by DeepPolyR at [0, 8]=[-1.688307, 1.285591]
(DeepPolyR) Proved 4 6 8 by DeepPolyR at [0, 8]=[-1.688307, -0.201358]
(DeepPolyR) Proved 4 6 8 by DeepPolyR at [8, 8]=[1.285591, 1.285591]
(Overall) Proved 4 6 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 9 by DeepPolyR at [0, 8]=[-1.394015, 1.579883]
(DeepPolyR) Proved 4 6 9 by DeepPolyR at [0, 8]=[-1.394015, -1.394015]
(DeepPolyR) Proved 4 6 9 by DeepPolyR at [1, 8]=[0.092934, 1.579883]
(Overall) Proved 4 6 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 7 0 by DeepPolyR at [0, 8]=[-1.270102, 1.703796]
(DeepPolyR) Proved 4 7 0 by DeepPolyR at [0, 8]=[-1.270102, -1.270102]
(DeepPolyR) Fail to prove 4 7 0 by DeepPolyR at [1, 8]=[0.216847, 1.703796]
(Divide and Counter) Proved 4 7 0 by DeepPoly at [0]=-1.270102
(Divide and Counter) Fail to prove 4 7 0 by DeepPolyR at [1, 8]=[0.216847, 1.703796]
(Divide and Counter) Proved 4 7 0 by DeepPolyR at [1, 7]=[0.216847, 0.960321]
(Divide and Counter) Fail to prove 4 7 0 by DeepPoly at [8]=1.703796
(Overall) Fail to prove 4 7 0 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 4 7 1 by DeepPolyR at [0, 8]=[-1.796730, 1.177168]
(DeepPolyR) Proved 4 7 1 by DeepPolyR at [0, 8]=[-1.796730, -1.796730]
(DeepPolyR) Proved 4 7 1 by DeepPolyR at [1, 8]=[0.061956, 1.177168]
(Overall) Proved 4 7 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 7 2 by DeepPolyR at [0, 8]=[-1.115212, 1.858686]
(DeepPolyR) Proved 4 7 2 by DeepPolyR at [0, 8]=[-1.115212, -0.061956]
(DeepPolyR) Fail to prove 4 7 2 by DeepPolyR at [8, 8]=[1.858686, 1.858686]
(Divide and Counter) Proved 4 7 2 by DeepPolyR at [0, 7]=[-1.115212, -0.061956]
(Divide and Counter) Fail to prove 4 7 2 by DeepPoly at [8]=1.858686
(Overall) Fail to prove 4 7 2 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 4 7 3 by DeepPolyR at [0, 8]=[-1.781241, 1.192657]
(DeepPolyR) Proved 4 7 3 by DeepPolyR at [0, 8]=[-1.781241, -0.294292]
(DeepPolyR) Proved 4 7 3 by DeepPolyR at [8, 8]=[1.192657, 1.192657]
(Overall) Proved 4 7 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 7 4 by DeepPolyR at [0, 8]=[-1.053256, 1.920643]
(DeepPolyR) Proved 4 7 4 by DeepPolyR at [0, 8]=[-1.053256, 0.433693]
(DeepPolyR) Proved 4 7 4 by DeepPolyR at [2, 8]=[0.681518, 1.920643]
(Overall) Proved 4 7 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 7 5 by DeepPolyR at [0, 8]=[-1.734774, 1.239124]
(DeepPolyR) Proved 4 7 5 by DeepPolyR at [0, 8]=[-1.734774, -1.734774]
(DeepPolyR) Proved 4 7 5 by DeepPolyR at [1, 8]=[0.000000, 1.239124]
(Overall) Proved 4 7 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 7 6 by DeepPolyR at [0, 8]=[-1.920643, 1.053256]
(DeepPolyR) Proved 4 7 6 by DeepPolyR at [0, 8]=[-1.920643, -0.433693]
(DeepPolyR) Proved 4 7 6 by DeepPolyR at [8, 8]=[1.053256, 1.053256]
(Overall) Proved 4 7 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 7 7 by DeepPolyR at [0, 8]=[-1.533416, 1.440482]
(DeepPolyR) Proved 4 7 7 by DeepPolyR at [0, 8]=[-1.533416, -1.533416]
(DeepPolyR) Fail to prove 4 7 7 by DeepPolyR at [1, 8]=[0.201358, 1.440482]
(Divide and Counter) Proved 4 7 7 by DeepPoly at [0]=-1.533416
(Divide and Counter) Fail to prove 4 7 7 by DeepPolyR at [1, 8]=[0.201358, 1.440482]
(Divide and Counter) Proved 4 7 7 by DeepPolyR at [1, 6]=[0.201358, 0.480161]
(Divide and Counter) Fail to prove 4 7 7 by DeepPolyR at [7, 8]=[0.944832, 1.440482]
(Divide and Counter) Proved 4 7 7 by DeepPoly at [7]=0.944832
(Divide and Counter) Fail to prove 4 7 7 by DeepPoly at [8]=1.440482
(Overall) Fail to prove 4 7 7 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 4 7 8 by DeepPolyR at [0, 8]=[-1.874175, 1.099723]
(DeepPolyR) Proved 4 7 8 by DeepPolyR at [0, 8]=[-1.874175, -1.874175]
(DeepPolyR) Proved 4 7 8 by DeepPolyR at [1, 8]=[0.046467, 1.099723]
(Overall) Proved 4 7 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 7 9 by DeepPolyR at [0, 8]=[-1.765752, 1.208146]
(DeepPolyR) Proved 4 7 9 by DeepPolyR at [0, 8]=[-1.765752, -0.278803]
(DeepPolyR) Proved 4 7 9 by DeepPolyR at [8, 8]=[1.208146, 1.208146]
(Overall) Proved 4 7 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 0 by DeepPolyR at [0, 8]=[-1.703796, 1.270102]
(DeepPolyR) Proved 4 8 0 by DeepPolyR at [0, 8]=[-1.703796, -1.703796]
(DeepPolyR) Proved 4 8 0 by DeepPolyR at [1, 8]=[0.030978, 1.270102]
(Overall) Proved 4 8 0 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 1 by DeepPolyR at [0, 8]=[-1.022278, 1.951621]
(DeepPolyR) Proved 4 8 1 by DeepPolyR at [0, 8]=[-1.022278, -0.015489]
(DeepPolyR) Proved 4 8 1 by DeepPolyR at [8, 8]=[1.951621, 1.951621]
(Overall) Proved 4 8 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 2 by DeepPolyR at [0, 8]=[-1.626351, 1.347548]
(DeepPolyR) Proved 4 8 2 by DeepPolyR at [0, 8]=[-1.626351, -0.511139]
(DeepPolyR) Proved 4 8 2 by DeepPolyR at [7, 8]=[-0.139401, 1.347548]
(Overall) Proved 4 8 2 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 3 by DeepPolyR at [0, 8]=[-1.905154, 1.068745]
(DeepPolyR) Proved 4 8 3 by DeepPolyR at [0, 8]=[-1.905154, -1.905154]
(DeepPolyR) Proved 4 8 3 by DeepPolyR at [1, 8]=[0.015489, 1.068745]
(Overall) Proved 4 8 3 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 4 by DeepPolyR at [0, 8]=[-1.796730, 1.177168]
(DeepPolyR) Proved 4 8 4 by DeepPolyR at [0, 8]=[-1.796730, -1.796730]
(DeepPolyR) Proved 4 8 4 by DeepPolyR at [1, 8]=[0.061956, 1.177168]
(Overall) Proved 4 8 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 5 by DeepPolyR at [0, 8]=[-1.208146, 1.765752]
(DeepPolyR) Proved 4 8 5 by DeepPolyR at [0, 8]=[-1.208146, -0.092934]
(DeepPolyR) Proved 4 8 5 by DeepPolyR at [8, 8]=[1.765752, 1.765752]
(Overall) Proved 4 8 5 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 6 by DeepPolyR at [0, 8]=[-1.688307, 1.285591]
(DeepPolyR) Proved 4 8 6 by DeepPolyR at [0, 8]=[-1.688307, -1.688307]
(DeepPolyR) Proved 4 8 6 by DeepPolyR at [1, 8]=[0.046467, 1.285591]
(Overall) Proved 4 8 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 7 by DeepPolyR at [0, 8]=[-1.301080, 1.672818]
(DeepPolyR) Proved 4 8 7 by DeepPolyR at [0, 8]=[-1.301080, -0.061956]
(DeepPolyR) Proved 4 8 7 by DeepPolyR at [8, 8]=[1.672818, 1.672818]
(Overall) Proved 4 8 7 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 8 by DeepPolyR at [0, 8]=[-1.889665, 1.084234]
(DeepPolyR) Proved 4 8 8 by DeepPolyR at [0, 8]=[-1.889665, -1.889665]
(DeepPolyR) Proved 4 8 8 by DeepPolyR at [1, 8]=[0.030978, 1.084234]
(Overall) Proved 4 8 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 9 by DeepPolyR at [0, 8]=[-1.750263, 1.223635]
(DeepPolyR) Proved 4 8 9 by DeepPolyR at [0, 8]=[-1.750263, -1.750263]
(DeepPolyR) Proved 4 8 9 by DeepPolyR at [1, 8]=[0.108423, 1.223635]
(Overall) Proved 4 8 9 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 9 0 by DeepPolyR at [0, 8]=[-1.688307, 1.285591]
(DeepPolyR) Proved 4 9 0 by DeepPolyR at [0, 8]=[-1.688307, -1.688307]
(DeepPolyR) Fail to prove 4 9 0 by DeepPolyR at [1, 8]=[0.046467, 1.285591]
(Divide and Counter) Proved 4 9 0 by DeepPoly at [0]=-1.688307
(Divide and Counter) Fail to prove 4 9 0 by DeepPolyR at [1, 8]=[0.046467, 1.285591]
(Divide and Counter) Proved 4 9 0 by DeepPolyR at [1, 6]=[0.046467, 0.418204]
(Divide and Counter) Fail to prove 4 9 0 by DeepPolyR at [7, 8]=[0.789942, 1.285591]
(Divide and Counter) Fail to prove 4 9 0 by DeepPoly at [7]=0.789942
(Overall) Fail to prove 4 9 0 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 4 9 1 by DeepPolyR at [0, 8]=[-1.920643, 1.053256]
(DeepPolyR) Proved 4 9 1 by DeepPolyR at [0, 8]=[-1.920643, -1.920643]
(DeepPolyR) Proved 4 9 1 by DeepPolyR at [1, 8]=[0.000000, 1.053256]
(Overall) Proved 4 9 1 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 9 2 by DeepPolyR at [0, 8]=[-1.579883, 1.394015]
(DeepPolyR) Proved 4 9 2 by DeepPolyR at [0, 8]=[-1.579883, -1.579883]
(DeepPolyR) Fail to prove 4 9 2 by DeepPolyR at [1, 8]=[0.154891, 1.394015]
(Divide and Counter) Proved 4 9 2 by DeepPoly at [0]=-1.579883
(Divide and Counter) Fail to prove 4 9 2 by DeepPolyR at [1, 8]=[0.154891, 1.394015]
(Divide and Counter) Proved 4 9 2 by DeepPolyR at [1, 6]=[0.154891, 0.464672]
(Divide and Counter) Fail to prove 4 9 2 by DeepPolyR at [7, 8]=[0.898365, 1.394015]
(Divide and Counter) Fail to prove 4 9 2 by DeepPoly at [7]=0.898365
(Overall) Fail to prove 4 9 2 with all masks. Summary: 0 1 0 

(Divide and Counter) Fail to prove 4 9 3 by DeepPolyR at [0, 8]=[-1.270102, 1.703796]
(DeepPolyR) Proved 4 9 3 by DeepPolyR at [0, 8]=[-1.270102, -0.030978]
(DeepPolyR) Fail to prove 4 9 3 by DeepPolyR at [8, 8]=[1.703796, 1.703796]
(Divide and Counter) Proved 4 9 3 by DeepPolyR at [0, 7]=[-1.270102, -0.030978]
(Divide and Counter) Fail to prove 4 9 3 by DeepPoly at [8]=1.703796
(Overall) Fail to prove 4 9 3 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 4 9 4 by DeepPolyR at [0, 8]=[-1.626351, 1.347548]
(DeepPolyR) Proved 4 9 4 by DeepPolyR at [0, 8]=[-1.626351, -0.511139]
(DeepPolyR) Proved 4 9 4 by DeepPolyR at [7, 8]=[-0.139401, 1.347548]
(Overall) Proved 4 9 4 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 9 5 by DeepPolyR at [0, 8]=[-1.765752, 1.208146]
(DeepPolyR) Proved 4 9 5 by DeepPolyR at [0, 8]=[-1.765752, -1.765752]
(DeepPolyR) Fail to prove 4 9 5 by DeepPolyR at [1, 8]=[0.092934, 1.208146]
(Divide and Counter) Proved 4 9 5 by DeepPoly at [0]=-1.765752
(Divide and Counter) Fail to prove 4 9 5 by DeepPolyR at [1, 8]=[0.092934, 1.208146]
(Divide and Counter) Proved 4 9 5 by DeepPolyR at [1, 6]=[0.092934, 0.464672]
(Divide and Counter) Fail to prove 4 9 5 by DeepPolyR at [7, 8]=[0.712496, 1.208146]
(Divide and Counter) Proved 4 9 5 by DeepPoly at [7]=0.712496
(Divide and Counter) Fail to prove 4 9 5 by DeepPoly at [8]=1.208146
(Overall) Fail to prove 4 9 5 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 4 9 6 by DeepPolyR at [0, 8]=[-1.951621, 1.022278]
(DeepPolyR) Proved 4 9 6 by DeepPolyR at [0, 8]=[-1.951621, -0.712496]
(DeepPolyR) Proved 4 9 6 by DeepPolyR at [7, 8]=[-0.464672, 1.022278]
(Overall) Proved 4 9 6 with all masks. Summary: 1 1 1 

(Divide and Counter) Fail to prove 4 9 7 by DeepPolyR at [0, 8]=[-1.378526, 1.595373]
(DeepPolyR) Proved 4 9 7 by DeepPolyR at [0, 8]=[-1.378526, 0.108423]
(DeepPolyR) Fail to prove 4 9 7 by DeepPolyR at [2, 8]=[0.542117, 1.595373]
(Divide and Counter) Proved 4 9 7 by DeepPolyR at [0, 1]=[-1.378526, 0.108423]
(Divide and Counter) Fail to prove 4 9 7 by DeepPolyR at [2, 8]=[0.542117, 1.595373]
(Divide and Counter) Fail to prove 4 9 7 by DeepPolyR at [2, 7]=[0.542117, 0.851898]
(Divide and Counter) Proved 4 9 7 by DeepPolyR at [2, 5]=[0.542117, 0.604073]
(Divide and Counter) Fail to prove 4 9 7 by DeepPolyR at [6, 7]=[0.727986, 0.851898]
(Divide and Counter) Proved 4 9 7 by DeepPoly at [6]=0.727986
(Divide and Counter) Fail to prove 4 9 7 by DeepPoly at [7]=0.851898
(Overall) Fail to prove 4 9 7 with all masks. Summary: 0 1 0 

(Divide and Counter) Proved 4 9 8 by DeepPolyR at [0, 8]=[-1.409504, 1.564394]
(DeepPolyR) Proved 4 9 8 by DeepPolyR at [0, 8]=[-1.409504, -0.170380]
(DeepPolyR) Proved 4 9 8 by DeepPolyR at [8, 8]=[1.564394, 1.564394]
(Overall) Proved 4 9 8 with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 9 9 by DeepPolyR at [0, 8]=[-1.610862, 1.363037]
(DeepPolyR) Proved 4 9 9 by DeepPolyR at [0, 8]=[-1.610862, -1.610862]
(DeepPolyR) Proved 4 9 9 by DeepPolyR at [1, 8]=[0.123912, 1.363037]
(Overall) Proved 4 9 9 with all masks. Summary: 1 1 1 

Elapsed Time: 2.445
flg =1
all_proved =0
queries = 376
GPUPoly 0.13.0S Debug (built Apr 26 2024 13:58:03) - Copyright (C) 2020 Department of Computer Science, ETH Zurich.
This program comes with ABSOLUTELY NO WARRANTY. This is free software, and you are welcome to redistribute it and to modify it under the terms of the GNU LGPLv3.

Warning: bit-flip at the first layer is not supported due to heavy technical debt (CPU version is supported)
Validating Bias
weights_path: benchmark/benchmark_QAT/QAT_mnist_3blk_10_10_10_qu_8.h5
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
label is 4
bit_all is 8
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
(Divide and Counter) Proved 2 0 (bias) by DeepPolyR at [0, 8]=[-0.891088, 1.145685]
(DeepPolyR) Proved 2 0 (bias) by DeepPolyR at [0, 8]=[-0.891088, 0.127298]
(DeepPolyR) Proved 2 0 (bias) by DeepPolyR at [2, 8]=[0.381895, 1.145685]
(Overall) Proved 2 0 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 1 (bias) by DeepPolyR at [0, 8]=[-1.060819, 0.975954]
(DeepPolyR) Proved 2 1 (bias) by DeepPolyR at [0, 8]=[-1.060819, -1.060819]
(DeepPolyR) Proved 2 1 (bias) by DeepPolyR at [1, 8]=[0.127298, 0.975954]
(Overall) Proved 2 1 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 2 (bias) by DeepPolyR at [0, 8]=[-1.060819, 0.975954]
(DeepPolyR) Proved 2 2 (bias) by DeepPolyR at [0, 8]=[-1.060819, -1.060819]
(DeepPolyR) Proved 2 2 (bias) by DeepPolyR at [1, 8]=[0.127298, 0.975954]
(Overall) Proved 2 2 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 3 (bias) by DeepPolyR at [0, 8]=[-1.082036, 0.954737]
(DeepPolyR) Proved 2 3 (bias) by DeepPolyR at [0, 8]=[-1.082036, -1.082036]
(DeepPolyR) Proved 2 3 (bias) by DeepPolyR at [1, 8]=[0.106082, 0.954737]
(Overall) Proved 2 3 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 4 (bias) by DeepPolyR at [0, 8]=[-0.891088, 1.145685]
(DeepPolyR) Proved 2 4 (bias) by DeepPolyR at [0, 8]=[-0.891088, 0.127298]
(DeepPolyR) Proved 2 4 (bias) by DeepPolyR at [2, 8]=[0.381895, 1.145685]
(Overall) Proved 2 4 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 5 (bias) by DeepPolyR at [0, 8]=[-0.954737, 1.082036]
(DeepPolyR) Proved 2 5 (bias) by DeepPolyR at [0, 8]=[-0.954737, 0.063649]
(DeepPolyR) Proved 2 5 (bias) by DeepPolyR at [2, 8]=[0.360679, 1.082036]
(Overall) Proved 2 5 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 6 (bias) by DeepPolyR at [0, 8]=[-0.997170, 1.039603]
(DeepPolyR) Proved 2 6 (bias) by DeepPolyR at [0, 8]=[-0.997170, 0.021216]
(DeepPolyR) Proved 2 6 (bias) by DeepPolyR at [2, 8]=[0.339462, 1.039603]
(Overall) Proved 2 6 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 7 (bias) by DeepPolyR at [0, 8]=[-0.944129, 1.092644]
(DeepPolyR) Proved 2 7 (bias) by DeepPolyR at [0, 8]=[-0.944129, 0.074257]
(DeepPolyR) Proved 2 7 (bias) by DeepPolyR at [2, 8]=[0.371287, 1.092644]
(Overall) Proved 2 7 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 8 (bias) by DeepPolyR at [0, 8]=[-0.997170, 1.039603]
(DeepPolyR) Proved 2 8 (bias) by DeepPolyR at [0, 8]=[-0.997170, 0.021216]
(DeepPolyR) Proved 2 8 (bias) by DeepPolyR at [2, 8]=[0.339462, 1.039603]
(Overall) Proved 2 8 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 2 9 (bias) by DeepPolyR at [0, 8]=[-0.965346, 1.071428]
(DeepPolyR) Proved 2 9 (bias) by DeepPolyR at [0, 8]=[-0.965346, 0.053041]
(DeepPolyR) Proved 2 9 (bias) by DeepPolyR at [2, 8]=[0.350070, 1.071428]
(Overall) Proved 2 9 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 0 (bias) by DeepPolyR at [0, 8]=[-1.279980, 1.227736]
(DeepPolyR) Proved 3 0 (bias) by DeepPolyR at [0, 8]=[-1.279980, -1.279980]
(DeepPolyR) Proved 3 0 (bias) by DeepPolyR at [1, 8]=[0.182854, 1.227736]
(Overall) Proved 3 0 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 1 (bias) by DeepPolyR at [0, 8]=[-1.306103, 1.201614]
(DeepPolyR) Proved 3 1 (bias) by DeepPolyR at [0, 8]=[-1.306103, -1.306103]
(DeepPolyR) Proved 3 1 (bias) by DeepPolyR at [1, 8]=[0.156732, 1.201614]
(Overall) Proved 3 1 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 2 (bias) by DeepPolyR at [0, 8]=[-1.306103, 1.201614]
(DeepPolyR) Proved 3 2 (bias) by DeepPolyR at [0, 8]=[-1.306103, -1.306103]
(DeepPolyR) Proved 3 2 (bias) by DeepPolyR at [1, 8]=[0.156732, 1.201614]
(Overall) Proved 3 2 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 3 (bias) by DeepPolyR at [0, 8]=[-1.240797, 1.266919]
(DeepPolyR) Proved 3 3 (bias) by DeepPolyR at [0, 8]=[-1.240797, -1.240797]
(DeepPolyR) Proved 3 3 (bias) by DeepPolyR at [1, 8]=[0.013061, 1.266919]
(Overall) Proved 3 3 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 4 (bias) by DeepPolyR at [0, 8]=[-1.436713, 1.071004]
(DeepPolyR) Proved 3 4 (bias) by DeepPolyR at [0, 8]=[-1.436713, -1.436713]
(DeepPolyR) Proved 3 4 (bias) by DeepPolyR at [1, 8]=[0.026122, 1.071004]
(Overall) Proved 3 4 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 5 (bias) by DeepPolyR at [0, 8]=[-1.136309, 1.371408]
(DeepPolyR) Proved 3 5 (bias) by DeepPolyR at [0, 8]=[-1.136309, -1.136309]
(DeepPolyR) Proved 3 5 (bias) by DeepPolyR at [1, 8]=[0.117549, 1.371408]
(Overall) Proved 3 5 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 6 (bias) by DeepPolyR at [0, 8]=[-1.175492, 1.332225]
(DeepPolyR) Proved 3 6 (bias) by DeepPolyR at [0, 8]=[-1.175492, -1.175492]
(DeepPolyR) Proved 3 6 (bias) by DeepPolyR at [1, 8]=[0.078366, 1.332225]
(Overall) Proved 3 6 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 7 (bias) by DeepPolyR at [0, 8]=[-1.319164, 1.188553]
(DeepPolyR) Proved 3 7 (bias) by DeepPolyR at [0, 8]=[-1.319164, -1.319164]
(DeepPolyR) Proved 3 7 (bias) by DeepPolyR at [1, 8]=[0.143671, 1.188553]
(Overall) Proved 3 7 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 8 (bias) by DeepPolyR at [0, 8]=[-1.502018, 1.005699]
(DeepPolyR) Proved 3 8 (bias) by DeepPolyR at [0, 8]=[-1.502018, -1.502018]
(DeepPolyR) Proved 3 8 (bias) by DeepPolyR at [1, 8]=[0.065305, 1.005699]
(Overall) Proved 3 8 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 3 9 (bias) by DeepPolyR at [0, 8]=[-1.319164, 1.188553]
(DeepPolyR) Proved 3 9 (bias) by DeepPolyR at [0, 8]=[-1.319164, -1.319164]
(DeepPolyR) Proved 3 9 (bias) by DeepPolyR at [1, 8]=[0.143671, 1.188553]
(Overall) Proved 3 9 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 0 (bias) by DeepPolyR at [0, 8]=[-1.688307, 1.285591]
(DeepPolyR) Proved 4 0 (bias) by DeepPolyR at [0, 8]=[-1.688307, -1.688307]
(DeepPolyR) Proved 4 0 (bias) by DeepPolyR at [1, 8]=[0.046467, 1.285591]
(Overall) Proved 4 0 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 1 (bias) by DeepPolyR at [0, 8]=[-1.812219, 1.161679]
(DeepPolyR) Proved 4 1 (bias) by DeepPolyR at [0, 8]=[-1.812219, -1.812219]
(DeepPolyR) Proved 4 1 (bias) by DeepPolyR at [1, 8]=[0.046467, 1.161679]
(Overall) Proved 4 1 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 2 (bias) by DeepPolyR at [0, 8]=[-1.874175, 1.099723]
(DeepPolyR) Proved 4 2 (bias) by DeepPolyR at [0, 8]=[-1.874175, -1.874175]
(DeepPolyR) Proved 4 2 (bias) by DeepPolyR at [1, 8]=[0.046467, 1.099723]
(Overall) Proved 4 2 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 3 (bias) by DeepPolyR at [0, 8]=[-1.657329, 1.316570]
(DeepPolyR) Proved 4 3 (bias) by DeepPolyR at [0, 8]=[-1.657329, -1.657329]
(DeepPolyR) Proved 4 3 (bias) by DeepPolyR at [1, 8]=[0.077445, 1.316570]
(Overall) Proved 4 3 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 4 (bias) by DeepPolyR at [0, 8]=[-1.672818, 1.301080]
(DeepPolyR) Proved 4 4 (bias) by DeepPolyR at [0, 8]=[-1.672818, -1.672818]
(DeepPolyR) Proved 4 4 (bias) by DeepPolyR at [1, 8]=[0.061956, 1.301080]
(Overall) Proved 4 4 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 5 (bias) by DeepPolyR at [0, 8]=[-1.626351, 1.347548]
(DeepPolyR) Proved 4 5 (bias) by DeepPolyR at [0, 8]=[-1.626351, -1.626351]
(DeepPolyR) Proved 4 5 (bias) by DeepPolyR at [1, 8]=[0.108423, 1.347548]
(Overall) Proved 4 5 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 6 (bias) by DeepPolyR at [0, 8]=[-1.750263, 1.223635]
(DeepPolyR) Proved 4 6 (bias) by DeepPolyR at [0, 8]=[-1.750263, -1.750263]
(DeepPolyR) Proved 4 6 (bias) by DeepPolyR at [1, 8]=[0.108423, 1.223635]
(Overall) Proved 4 6 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 7 (bias) by DeepPolyR at [0, 8]=[-1.781241, 1.192657]
(DeepPolyR) Proved 4 7 (bias) by DeepPolyR at [0, 8]=[-1.781241, -1.781241]
(DeepPolyR) Proved 4 7 (bias) by DeepPolyR at [1, 8]=[0.077445, 1.192657]
(Overall) Proved 4 7 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 8 (bias) by DeepPolyR at [0, 8]=[-1.812219, 1.161679]
(DeepPolyR) Proved 4 8 (bias) by DeepPolyR at [0, 8]=[-1.812219, -1.812219]
(DeepPolyR) Proved 4 8 (bias) by DeepPolyR at [1, 8]=[0.046467, 1.161679]
(Overall) Proved 4 8 (bias) with all masks. Summary: 1 1 1 

(Divide and Counter) Proved 4 9 (bias) by DeepPolyR at [0, 8]=[-1.641840, 1.332059]
(DeepPolyR) Proved 4 9 (bias) by DeepPolyR at [0, 8]=[-1.641840, -1.641840]
(DeepPolyR) Proved 4 9 (bias) by DeepPolyR at [1, 8]=[0.092934, 1.332059]
(Overall) Proved 4 9 (bias) with all masks. Summary: 1 1 1 

Elapsed Time: 0.224
flg =0
all_proved =1
queries = 30
