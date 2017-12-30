# Neural Network Quantization


## 1. Gemmlowp Quantization
### Basic idea:
Quantization as an affine map.

### Formula:
```
real_value = C * (quantized_value + D)
real_value = scale * (quantized_value - zero_point)
int32_accumulator =
    Sum_over_i(
      lhs_quantized_value[i] *
      rhs_quantized_value[i]
    )      

ON-DEVICE RUNTIME QUANTIZED CODE:
result_quantized_value = result_zero_point +
    (lhs_scale * rhs_scale / result_scale) * int32_accumulator 
    
```                                               
### Description: 
https://github.com/google/gemmlowp/blob/master/doc/quantization.md
### Implementation code:
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/kernels

https://github.com/google/gemmlowp
https://github.com/google/gemmlowp/blob/master/doc/quantization_example.cc

### Quantized model:
#### quantized input
![image](https://github.com/shengweigit/notebook/blob/master/compression/quantization/png/quantized_input.png)

#### quantized conv2d
![image](https://github.com/shengweigit/notebook/blob/master/compression/quantization/png/quantized_conv2d.png)

#### quantized bias
![image](https://github.com/shengweigit/notebook/blob/master/compression/quantization/png/quantized_bias.png)

#### quantized add bias
![image](https://github.com/shengweigit/notebook/blob/master/compression/quantization/png/quantized_bias_add.png)

#### quantized relu
![image](https://github.com/shengweigit/notebook/blob/master/compression/quantization/png/quantized_relu.png)

#### quantized maxpool
![image](https://github.com/shengweigit/notebook/blob/master/compression/quantization/png/quantized_maxpool.png)

### Experiment results:
