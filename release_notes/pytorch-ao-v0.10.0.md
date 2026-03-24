# pytorch/ao v0.10.0

Source: https://github.com/pytorch/ao/releases/tag/v0.10.0

*(Features, highlights, backwards-incompatible changes, deprecations, and new features only.)*

## Highlights

We are excited to announce the 0.10.0 release of torchao! This release adds support for end to end training for mxfp8 on Nvidia B200, PARQ (for quantization aware training), module swap quantization API to for research, and some updates for low bit kernels!

### Low Bit Optimizers moved to Official Support ([https://github.com/pytorch/ao/pull/1864](https://github.com/pytorch/ao/pull/1864))

[Low bit optimizers](https://github.com/pytorch/ao/releases/tag/v0.4.0) (added in 0.4) is moved out of prototype and now have official support in torchao.

### \[Prototype\] End to End Training Support for mxfp8 on NVIDIA B200 (\#1786, \#1841, \#1951, \#1932, \#1980)

We have an early version of the end to end training workflow for the [mxfp8](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) dtypes with torch.compile on NVIDIA B200, with the cuBLAS mxfp8 gemm seeing an observed speedup of **over 2x** over bfloat16 gemm, and casts from bfloat16 to mxfp8 achieving **up to 5.5 TB/s**. Please see our [README.md for MX](https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/README.md) for more information. We plan to improve performance further in future releases.

### \[Prototype\] Piecewise-Affine Regularized Quantization ([https://github.com/pytorch/ao/pull/1738](https://github.com/pytorch/ao/pull/1738)) 

* [PARQ](https://github.com/facebookresearch/parq) is a [new theoretical framework](https://arxiv.org/abs/2503.15748) for inducing quantization through regularization. It supports standard QAT, as well as new gradual quantization methods, in an easy to use [optimizer-only interface](https://github.com/facebookresearch/parq). No modifications to a model’s forward or backward pass are needed for quantization.

```py
from torchao.prototype.parq.optim import QuantOptimizer, ProxHardQuant
from torchao.prototype.parq.quant import UnifQuantizer

# Separate quantizable from non-quantizable parameter groups
param_groups = [
 {"params": weights, "quant_bits": 2}, # add extra quant_bits key for QAT
 {"params": others},
]

# Initialize any torch.optim.Optimizer
base_optimizer = torch.optim.SGD(param_groups, lr=0.1, momentum=0.9, weight_decay=1e-4)

# Apply a simple wrapper to quantize in optimizer.step()
optimizer = QuantOptimizer(
 base_optimizer, quantizer=UnifQuantizer(), prox_map=ProxHardQuant()
)
```

### \[Prototype\] Module Swap Quantization API ([https://github.com/pytorch/ao/pull/1886](https://github.com/pytorch/ao/pull/1886))

We added a prototype API for post-training quantization. Users can swap their linear or embedding layers into their QuantizedLinear and QuantizedEmbedding counterparts, and set the quantizers that specify how they want the input activations or weights to be quantized:

```py
quantized_linear = QuantizedLinear(...)
quantized_linear.weight_quantization = IntQuantizer(
 num_bits=4,
 group_size=32,
 dynamic=True,
 quantization_mode="symmetric",
)
quantized_linear.input_quantization = CodeBookQuantizer(
 num_bits=8,
 features=10,
)
```

Note: The API is highly subject to change and will be integrated with quantize\_ in the future. For more detail, please see the [README](https://github.com/pytorch/ao/blob/v0.10.0-rc1/torchao/prototype/quantization/module_swap/README.md).

### \[Prototype\] Low Bit Kernels Updates (\#1826, \#1935, \#1998, \#1652)

Low-bit CPU and MPS kernels are now pip installable from source. To install torchao with low-bit CPU kernels, you can use the following command on an Arm-based Mac: 

```
USE_CPP=1 pip install git+https://github.com/pytorch/ao.git
```

You can then quantize your model to run on Arm-based Macs with high-performance CPU kernels in torchao. `SharedEmbeddingQuantizer,EmbeddingQuantizer`, and `Int8DynamicActivationIntxWeightConfig` all support 1-8 bit quantization. 

```py
from torchao.experimental.quant_api import Int8DynamicActivationIntxWeightConfig, SharedEmbeddingQuantizer, EmbeddingQuantizer
from torchao.quantization.granularity import PerGroup, PerRow
from torchao.quantization.quant_api import quantize_
# Quantize embedding/unembedding to 8-bits with SharedEmbeddingQuantizer 
# SharedEmbeddingQuantizer is for quantizing models like Llama1B/3B
# where the embedding/unembedding layers share weights 
# If the embedding/unembedding layers do not share weights, use 
# EmbeddingQuantizer instead 
SharedEmbeddingQuantizer(
	weight_dtype=torch.int8,
	granularity=PerRow(),
	has_weight_zeros=True
).quantize(model) # Quantize linear layers to 4-bits 
quantize_(
	model,
 Int8DynamicActivationIntxWeightConfig(
 weight_dtype=torch.int4,
	 granularity=PerGroup(128),
	 has_weight_zeros=False,
 )
)
```

---

## BC Breaking

### Delete delayed scaling from torchao.float8 ([https://github.com/pytorch/ao/pull/1753](https://github.com/pytorch/ao/pull/1753))

The following usage of \`Float8Config\` is deprecated in torchao v0.10.0:

```py
config = Float8LinearConfig(
 cast_config_input=CastConfig(scaling_type=ScalingType.DELAYED),
 cast_config_weight=CastConfig(scaling_type=ScalingType.DELAYED),
 cast_config_grad_output=CastConfig(scaling_type=ScalingType.DELAYED),
)
```

If you would like to use float8 training with delayed scaling, please use an earlier release of torchao. Please see [https://github.com/pytorch/ao/issues/1680](https://github.com/pytorch/ao/issues/1680) for more context about this deprecation.

### Enforce AOBaseConfig type in `quantize_`'s `config` argument ([https://github.com/pytorch/ao/pull/1861](https://github.com/pytorch/ao/pull/1861))

This was done following a deprecation window to simplify the arguments of quantize\_, please see [https://github.com/pytorch/ao/issues/1690](https://github.com/pytorch/ao/issues/1690) for more context.

```py
# torchao v.0.9.0
def quantize_( 
 model: torch.nn.Module,
 **config: Union[AOBaseConfig, Callable[[torch.nn.Module], torch.nn.Module]],** 
 filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
 set_inductor_config: Optional[bool] = None,
 device: Optional[torch.types.Device] = None,
):

# torchao v.0.10.0
def quantize_(
 model: torch.nn.Module,
 config: AOBaseConfig,
 filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
 set_inductor_config: Optional[bool] = None, 
 device: Optional[torch.types.Device] = None,
):
```
### Remove the `set_inductor_config` argument of `quantize_`. ([https://github.com/pytorch/ao/pull/1865](https://github.com/pytorch/ao/pull/1865))

This was done following a deprecation window to decouple quantize\_ from torchinductor, please see [https://github.com/pytorch/ao/issues/1715](https://github.com/pytorch/ao/issues/1715) for more context.

```py
# torchao v.0.9.0
def quantize_(
 ...,
 set_inductor_config: Optional[bool] = None,
 ...,
): 
 # if set_inductor_config != None, throw a deprecation warning
 # if set_inductor_config == None, set it to True to stay consistent with old behavior

# torchao v0.10.0
def quantize_(
 ...,
):
 # set_inductor_config is removed from quantize_ and moved to relevant individual workflows
```

---

## Deprecations

We removed some of our prototype features that are not used, including DORA ([https://github.com/pytorch/ao/pull/1815](https://github.com/pytorch/ao/pull/1815)), split\_k kernel ([https://github.com/pytorch/ao/pull/1816](https://github.com/pytorch/ao/pull/1816)), profiler ([https://github.com/pytorch/ao/pull/1862](https://github.com/pytorch/ao/pull/1862)) and bitnet ([https://github.com/pytorch/ao/pull/1866](https://github.com/pytorch/ao/pull/1866)).

---

## New Features

QAT

* Added PARQ ([https://github.com/pytorch/ao/pull/1738](https://github.com/pytorch/ao/pull/1738))

Low Bit Optimizers

* Promote Low Bit Optim out of prototype ([https://github.com/pytorch/ao/pull/1864](https://github.com/pytorch/ao/pull/1864))

Module swap quantization API

* Add module swap quantization API from Quanty ([https://github.com/pytorch/ao/pull/1886](https://github.com/pytorch/ao/pull/1886))

Benchmarking

* Micro-benchmark inference ([https://github.com/pytorch/ao/pull/1759](https://github.com/pytorch/ao/pull/1759)) 
* Add sparsity to benchmarking ([https://github.com/pytorch/ao/pull/1917](https://github.com/pytorch/ao/pull/1917)) 
* Add float8 training benchmarking scripts ([https://github.com/pytorch/ao/pull/1802](https://github.com/pytorch/ao/pull/1802))