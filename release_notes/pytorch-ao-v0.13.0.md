# pytorch/ao v0.13.0

Source: https://github.com/pytorch/ao/releases/tag/v0.13.0

**Note:** No `v0.13.0` release on GitHub; body is from [v0.13.0-rc8](https://github.com/pytorch/ao/releases/tag/v0.13.0-rc8) (final v0.13 RC).

*(Features, highlights, backwards-incompatible changes, deprecations, and new features only.)*

## **Highlights**

We are excited to announce the 0.13.0 release of torchao\! This release adds support for numerous QAT improvements, faster mxfp8 pretraining and more\!

### **Simpler Multi-step QAT API (**[https://github.com/pytorch/ao/pull/2629](https://github.com/pytorch/ao/pull/2629)**)**

We added a new, simpler, multi-step QAT API that uses only a single config. Now users can specify the target post-training quantization (PTQ) config as the base config and we will automatically infer the correct fake quantize configs to use\!

```py
from torchao.quantization import (
 quantize_,
 Int8DynamicActivationInt4WeightConfig
)
from torchao.quantization.qat import QATConfig

# prepare
base_config = Int8DynamicActivationInt4WeightConfig(group_size=32)
qat_config = QATConfig(base_config, step="prepare")
quantize_(m, qat_config)

# train (not shown)

# convert
quantize_(m, QATConfig(base_config, step="convert"))
```

For more advanced use cases, users can continue to specify specific FakeQuantizeConfigs as before:

```py
# prepare
activation_config = IntxFakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
weight_config = IntxFakeQuantizeConfig(torch.int4, group_size=32)
qat_config = QATConfig(
 activation_config=activation_config,
 weight_config=weight_config,
 step="prepare",
)
quantize_(model, qat_config)

# train and convert (not shown)
```

### **(Prototype) NVFP4 and FP8 QAT** ([https://github.com/pytorch/ao/pull/2735](https://github.com/pytorch/ao/pull/2735), [https://github.com/pytorch/ao/pull/2666](https://github.com/pytorch/ao/pull/2666))

We generalized QAT to support FP8 and NVFP4 use cases. You can try them out as follows:

```py
from torchao.quantization import (
 quantize_,
 Float8DynamicActivationInt4WeightConfig,
 Float8DynamicActivationFloat8WeightConfig,
 Float8WeightOnlyConfig,
)
from torchao.prototype.mx_formats import NVFP4InferenceConfig
from torchao.quantization.qat import QATConfig

# Pick a base config
base_config = Float8DynamicActivationInt4WeightConfig() # or
base_config = Float8DynamicActivationInt8WeightConfig() # or
base_config = NVFP4InferenceConfig()

# prepare
qat_config = QATConfig(base_config, step="prepare")
quantize_(m, qat_config)

# train (not shown)

# convert
quantize_(m, QATConfig(base_config, step="convert"))
```

Users can also use the more specific FakeQuantizeConfigs for more advanced use cases, e.g.:

```py
from torchao.quantization import PerRow
from torchao.quantization.qat import Float8FakeQuantizeConfig
from torchao.prototype.qat import NVFP4FakeQuantizeConfig

act_config = Float8FakeQuantizeConfig(torch.float8_e4m3fn, PerRow())
weight_config = NVFP4FakeQuantizeConfig(use_per_tensor_scale=True)

# prepare
qat_config = QATConfig(
 activation_config=activation_config,
 weight_config=weight_config,
 step="prepare",
)
quantize_(model, qat_config)

# train and convert (not shown)
```

### **(prototype) 1.2x MXFP8 dense pretraining speedups with torchtitan**

We landed performance improvements (such as a [faster to\_mx dim1 cast](https://github.com/pytorch/ao/pull/2513)) to our prototype MXFP8 training APIs, and we now achieve a **1.2x speedup vs bf16** on pretraining LLaMa 3 8B on NVIDIA B200. Please see our [training benchmarks README](https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats#training-e2e-benchmarks-on-nvidia-b200) for more information.

### **torchao float8 training now integrated into axolotl\!**

You can now use `torchao.float8` directly from [axolotl](https://github.com/axolotl-ai-cloud/axolotl) to achieve finetuning **QPS e2e speedups of up to 1.1x** on 3B parameter models ([docs](https://docs.axolotl.ai/docs/mixed_precision.html#sec-fp8), [release notes](https://github.com/axolotl-ai-cloud/axolotl/releases/tag/v0.12.0)).

## **BC Breaking**

**`Float8DynamicActivationFloat8WeightConfig` and `Float8WeightOnlyConfig` version bump to 2 ([https://github.com/pytorch/ao/pull/2650](https://github.com/pytorch/ao/pull/2650))** 

We updated the implementation for float8 Tensor, so bumps the default version from 1 to 2 for these two configs.

```
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "torchao-testing/opt-125m-Float8DynamicActivationFloat8WeightConfig-v1-0.13.dev"
quantized_model = AutoModelForCausalLM.from_pretrained(
 model_name,
 torch_dtype="bfloat16",
 device_map="cuda",
)

/data/users/jerryzh/ao/torchao/core/config.py:249: UserWarning: Stored version is not the same as current default version of the config: stored_version=1, current_version=2, please check the deprecation warning
 warnings.warn(
/data/users/jerryzh/ao/torchao/dtypes/floatx/float8_layout.py:113: UserWarning: Models quantized with version 1 of Float8DynamicActivationFloat8WeightConfig is deprecated and will no longer be supported in a future release, please upgrade torchao and quantize again, or download a newer torchao checkpoint, see https://github.com/pytorch/ao/issues/2649 for more details
 warnings.warn(
```

Suggestion: upgrade torchao to 0.13 and later and generate the checkpoint again:

```
quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
```

Or download the checkpoint again (please let us know if the checkpoint is not updated)

Please see [https://github.com/pytorch/ao/issues/2649](https://github.com/pytorch/ao/issues/2649) for more details around the deprecation.

### **QAT API Changes (**[https://github.com/pytorch/ao/pull/2628](https://github.com/pytorch/ao/pull/2628), [https://github.com/pytorch/ao/pull/2641](https://github.com/pytorch/ao/pull/2641))

On a high level, the following existing APIs are deprecated and replaced by these new ones. Although this is technically BC-breaking due to typing changes, it will not affect most users as old classes are kept around for now. They are planned to be removed in the next release, however.

```py
IntXQuantizationAwareTrainingConfig -> QATConfig
FromIntXQuantizationAwareTrainingConfig -> QATConfig
FakeQuantizeConfig -> IntxFakeQuantizeConfig
FakeQuantizer -> IntxFakeQuantizer
```

Please see [https://github.com/pytorch/ao/issues/2630](https://github.com/pytorch/ao/issues/2630) and the latest [QAT README](https://github.com/pytorch/ao/blob/main/torchao/quantization/qat/README.md#quantize_-api-recommended) for more information on how to migrate.

### **Remove old** `change_linear_weights_to_*` APIs ([https://github.com/pytorch/ao/pull/2721](https://github.com/pytorch/ao/pull/2721))

The following old quantization APIs no longer work and are removed:

```py
change_linear_weights_to_int8_dqtensors(model)
change_linear_weights_to_int8_woqtensors(model)
change_linear_weights_to_int4_woqtensors(model)
```

Please use the quantize\_ API with the following configs instead:

```py
quantize_(model, Int8WeightOnlyConfig())
quantize_(model, Int4WeightOnlyConfig())
```

## **Deprecations**

### **Deprecate old TORCH\_VERSION variables ([https://github.com/pytorch/ao/pull/2719](https://github.com/pytorch/ao/pull/2719))**

The following variables are deprecated and will be removed in the next release:

```py
TORCH_VERSION_AT_LEAST_2_2
TORCH_VERSION_AT_LEAST_2_3
TORCH_VERSION_AT_LEAST_2_4
TORCH_VERSION_AT_LEAST_2_5
TORCH_VERSION_AT_LEAST_2_6
TORCH_VERSION_AT_LEAST_2_7
TORCH_VERSION_AT_LEAST_2_8
TORCH_VERSION_AFTER_2_2
TORCH_VERSION_AFTER_2_3
TORCH_VERSION_AFTER_2_4
TORCH_VERSION_AFTER_2_5
```

### **Drop support for PyTorch 2.5 and before ([https://github.com/pytorch/ao/pull/2720](https://github.com/pytorch/ao/pull/2720))**

torchao only supports the latest 3 versions of PyTorch. Please upgrade to PyTorch 2.6.0+ if you were using an older version of PyTorch.

## **New Features**

* New multi-step QAT API ([https://github.com/pytorch/ao/pull/2629](https://github.com/pytorch/ao/pull/2629)) 
* Add float8 FakeQuantizeConfig and FakeQuantizer ([https://github.com/pytorch/ao/pull/2735](https://github.com/pytorch/ao/pull/2735)) 
* (prototype) Add NVFP4 QAT ([https://github.com/pytorch/ao/pull/2666](https://github.com/pytorch/ao/pull/2666))

## **Improvements**

* Add StretchedUnifTorchaoQuantizer ([https://github.com/pytorch/ao/pull/2576](https://github.com/pytorch/ao/pull/2576)) 
* Allow symmetric\_no\_clipping\_error for KleidiAI kernels, update Readme and validate Kleidi INT4 quantization path ([https://github.com/pytorch/ao/pull/2570](https://github.com/pytorch/ao/pull/2570)) 
* Enable powers of 2 cast in float8 rowwise\_with\_gw\_hp recipe ([https://github.com/pytorch/ao/pull/2677](https://github.com/pytorch/ao/pull/2677)) 
* Don't call erase if node is already erased in batch norm fusion. ([https://github.com/pytorch/ao/pull/2716](https://github.com/pytorch/ao/pull/2716)) 
* Generalize FakeQuantizer beyond intx ([https://github.com/pytorch/ao/pull/2714](https://github.com/pytorch/ao/pull/2714)) 
* Allow pattern replacement to ignore literals ([https://github.com/pytorch/ao/pull/2519](https://github.com/pytorch/ao/pull/2519)) 
* Replace `export_for_training` with `torch.export.export` ([https://github.com/pytorch/ao/pull/2724](https://github.com/pytorch/ao/pull/2724)) 
* Allow no quantization during QATConfig convert ([https://github.com/pytorch/ao/pull/2694](https://github.com/pytorch/ao/pull/2694)) 
* Int4 sparse marlin tensor ([https://github.com/pytorch/ao/pull/2771](https://github.com/pytorch/ao/pull/2771)) 
* Remove group\_size arg in Float8DynamicActivationInt4WeightConfig ([https://github.com/pytorch/ao/pull/2779](https://github.com/pytorch/ao/pull/2779)) 
* Fix batch norm folding in `prepare_pt2e` for multiple conv-\>BN chains sharing the same conv weights ([https://github.com/pytorch/ao/pull/2795](https://github.com/pytorch/ao/pull/2795)) 
* Add Float8Tensor ([https://github.com/pytorch/ao/pull/2463](https://github.com/pytorch/ao/pull/2463)) 
* (prototype) Allow per-group quantizers in QuantOptimizer, fix state\_dict ([https://github.com/pytorch/ao/pull/2743](https://github.com/pytorch/ao/pull/2743)) 
* (prototype) SpinQuant support split qkv (prototype) ([https://github.com/pytorch/ao/pull/2547](https://github.com/pytorch/ao/pull/2547)) 
* (prototype) Make AWQ more general ([https://github.com/pytorch/ao/pull/2400](https://github.com/pytorch/ao/pull/2400)) 
* (prototype) MX training 
 * Integration of new mxfp8 casting cuda kernel ([https://github.com/pytorch/ao/pull/2564](https://github.com/pytorch/ao/pull/2564)) 
 * Mx: expose scaling calculation methods in training UX ([https://github.com/pytorch/ao/pull/2620](https://github.com/pytorch/ao/pull/2620)) 
 * Mx: make CUDA kernel for dim1 cast in mxfp8\_cublas recipe ([https://github.com/pytorch/ao/pull/2661](https://github.com/pytorch/ao/pull/2661)) 
* (prototype) MoE training 
 * Mxfp8 emulated grouped gemm ([https://github.com/pytorch/ao/pull/2626](https://github.com/pytorch/ao/pull/2626)) 
 * Add differentiable mxfp8 grouped gemm with dynamic quant (forward pass) ([https://github.com/pytorch/ao/pull/2627](https://github.com/pytorch/ao/pull/2627)) 
 * Support for 2d-2d emulated mxfp8 grouped gemm ([https://github.com/pytorch/ao/pull/2632](https://github.com/pytorch/ao/pull/2632)) 
 * Backward pass for differentiable mxfp8 grouped gemm with dynamic quant ([https://github.com/pytorch/ao/pull/2639](https://github.com/pytorch/ao/pull/2639)) 
 * torch.compile support for ScaledGroupedMMTensor ([https://github.com/pytorch/ao/pull/2509](https://github.com/pytorch/ao/pull/2509)) 
 * Assert expert weights are column-major; preserve subclass with transpose ([https://github.com/pytorch/ao/pull/2663](https://github.com/pytorch/ao/pull/2663)) 
 * set token group alignment size to 16 for fp8 training test ([https://github.com/pytorch/ao/pull/2678](https://github.com/pytorch/ao/pull/2678)) 
 * Make scaling type configurable for MoE training ([https://github.com/pytorch/ao/pull/2642](https://github.com/pytorch/ao/pull/2642)) 
 * use smaller block sizes for per group scaling kernels to improve perf ([https://github.com/pytorch/ao/pull/2668](https://github.com/pytorch/ao/pull/2668)) 
 * add llama4 benchmarking script ([https://github.com/pytorch/ao/pull/2669](https://github.com/pytorch/ao/pull/2669)) 
 * add fp8 rowwise kernels for expert weights ([https://github.com/pytorch/ao/pull/2696](https://github.com/pytorch/ao/pull/2696)) 
 * add bench script for fp8 rowwise kernels and update autotune configs ([https://github.com/pytorch/ao/pull/2697](https://github.com/pytorch/ao/pull/2697)) 
 * integrate rowwise expert quant kernel ([https://github.com/pytorch/ao/pull/2698](https://github.com/pytorch/ao/pull/2698)) 
 * work around wrap\_triton bug by using normal custom ops instead for fp8 rowwise kernels ([https://github.com/pytorch/ao/pull/2734](https://github.com/pytorch/ao/pull/2734)) 
 * fix scaling type bug; refactor distributed tests ([https://github.com/pytorch/ao/pull/2749](https://github.com/pytorch/ao/pull/2749)) 
 * use llama4 shapes for kernel benchmarks ([https://github.com/pytorch/ao/pull/2756](https://github.com/pytorch/ao/pull/2756)) 
 * remove duplicate benchmark script ([https://github.com/pytorch/ao/pull/2762](https://github.com/pytorch/ao/pull/2762)) 
 * refactor to share benchmarking and profiling utils ([https://github.com/pytorch/ao/pull/2767](https://github.com/pytorch/ao/pull/2767)) 
 * add memory bandwidth calculations to kernel benchmarking scripts ([https://github.com/pytorch/ao/pull/2769](https://github.com/pytorch/ao/pull/2769)) 
 * update bench script to compare fp8 dynamic quant scaled\_grouped\_mm fwd+bwd against bf16 ([https://github.com/pytorch/ao/pull/2765](https://github.com/pytorch/ao/pull/2765)) 
* Float8 blockwise training (prototype) 
 * Add Triton kernels for fp8 blockwise quantization and GEMMs ([https://github.com/pytorch/ao/pull/2617](https://github.com/pytorch/ao/pull/2617)) 
 * Add Float8BlockwiseLinear for training ([https://github.com/pytorch/ao/pull/2618](https://github.com/pytorch/ao/pull/2618)) 
 * Improve fp8 blockwise gemm perf ([https://github.com/pytorch/ao/pull/2784](https://github.com/pytorch/ao/pull/2784))

## **Bug Fixes**

* Fix autocast handling for float8 training rowwise recipes ([https://github.com/pytorch/ao/pull/2587](https://github.com/pytorch/ao/pull/2587)) 
* NVFP4 \-\> Use more of e4m3 range for block\_scales ([https://github.com/pytorch/ao/pull/2604](https://github.com/pytorch/ao/pull/2604)) 
* Handle the case when param groups are passed to optimizer ([https://github.com/pytorch/ao/pull/2606](https://github.com/pytorch/ao/pull/2606)) 
* Fix bc breakage flex path ([https://github.com/pytorch/ao/pull/2652](https://github.com/pytorch/ao/pull/2652)) 
* Fix FSDP2 breakage in nightly ([https://github.com/pytorch/ao/pull/2684](https://github.com/pytorch/ao/pull/2684)) 
* When replacing literals with placeholders lists are always converted to ([https://github.com/pytorch/ao/pull/2518](https://github.com/pytorch/ao/pull/2518)) 
* Don't learn zero points for symmetric quantization ([https://github.com/pytorch/ao/pull/2739](https://github.com/pytorch/ao/pull/2739)) 
* fix ROCM build for newer hipblaslt BC-breaking change ([https://github.com/pytorch/ao/pull/2510](https://github.com/pytorch/ao/pull/2510)) 
* Fix missing QuantOptimizer methods ([https://github.com/pytorch/ao/pull/2770](https://github.com/pytorch/ao/pull/2770)) 
* Fix float8 \+ int4 QAT ([https://github.com/pytorch/ao/pull/2851](https://github.com/pytorch/ao/pull/2851)) 
* Allowlist WeightWithDynamicFloat8CastTensor for deserialization for checkpointing ([https://github.com/pytorch/ao/pull/2573](https://github.com/pytorch/ao/pull/2573))

## **Performance**

* Fix float8 rowwise inference perf with torch.compile ([https://github.com/pytorch/ao/pull/2672](https://github.com/pytorch/ao/pull/2672)) 
* Add CUDA kernel for MXFP8 dim1 casting ([https://github.com/pytorch/ao/pull/2513](https://github.com/pytorch/ao/pull/2513), [https://github.com/pytorch/ao/pull/2550](https://github.com/pytorch/ao/pull/2550)) 
* Extend the MX cast benchmark to include casting to mxfp4 ([https://github.com/pytorch/ao/pull/2693](https://github.com/pytorch/ao/pull/2693))

## **Documentation**

* Add QLoRA and FP8 to finetuning tutorial (part 2\) ([https://github.com/pytorch/ao/pull/2542](https://github.com/pytorch/ao/pull/2542)) 
* Clean up QAT API surface \+ add separate API ref ([https://github.com/pytorch/ao/pull/2567](https://github.com/pytorch/ao/pull/2567)) 
* Update float8 README with AMD MI300X benchmark results ([https://github.com/pytorch/ao/pull/2736](https://github.com/pytorch/ao/pull/2736)) 
* Update float8 [README.md](http://README.md) with more recent e2e performance numbers ([https://github.com/pytorch/ao/pull/2774](https://github.com/pytorch/ao/pull/2774), [https://github.com/pytorch/ao/pull/2580](https://github.com/pytorch/ao/pull/2580)) 
* Update quantization overview and contributor guide doc ([https://github.com/pytorch/ao/pull/2723](https://github.com/pytorch/ao/pull/2723)) 
* add e2e training benchmark results to mx\_formats README.md ([https://github.com/pytorch/ao/pull/2777](https://github.com/pytorch/ao/pull/2777)) 
* Update paper link readme ([https://github.com/pytorch/ao/pull/2563](https://github.com/pytorch/ao/pull/2563)) 
* Minor improvements to OpenVINOQuantizer ([https://github.com/pytorch/ao/pull/2581](https://github.com/pytorch/ao/pull/2581)) 
* Update README with PEFT integration \+ installation ([https://github.com/pytorch/ao/pull/2559](https://github.com/pytorch/ao/pull/2559))

## **Developers**

* Bump cutlass version to 4.1.0 ([https://github.com/pytorch/ao/pull/2589](https://github.com/pytorch/ao/pull/2589)) 
* Fix git repo url in citation ([https://github.com/pytorch/ao/pull/2599](https://github.com/pytorch/ao/pull/2599)) 
* Simplify Float8Linear ([https://github.com/pytorch/ao/pull/2594](https://github.com/pytorch/ao/pull/2594), [https://github.com/pytorch/ao/pull/2595](https://github.com/pytorch/ao/pull/2595)) 
* Convert quantization internal methods to private ([https://github.com/pytorch/ao/pull/2568](https://github.com/pytorch/ao/pull/2568)) 
* Reference representation of dqlinear int4 for xnnpack ([https://github.com/pytorch/ao/pull/2520](https://github.com/pytorch/ao/pull/2520)) 
* Refactors to align with new tensor subclass design 
 * Add all fbgemm kernel Tensors into Int4WeightOnlyConfig and Float8DynamicActivationInt4WeightConfig ([https://github.com/pytorch/ao/pull/2474](https://github.com/pytorch/ao/pull/2474)) 
 * Add support for float8 activation for Int4PreshuffledTensor ([https://github.com/pytorch/ao/pull/2437](https://github.com/pytorch/ao/pull/2437)) 
 * Align Int4Tensor implementation details with the design of Float8Tensor ([https://github.com/pytorch/ao/pull/2687](https://github.com/pytorch/ao/pull/2687)) 
 * Support `optional_tensor_names` in TorchAOBaseTensor ([https://github.com/pytorch/ao/pull/2710](https://github.com/pytorch/ao/pull/2710)) 
 * Update Int4PreshuffledTensor to align with implementation details of the Float8Tensor ([https://github.com/pytorch/ao/pull/2738](https://github.com/pytorch/ao/pull/2738)) 
 * Nvfp4 tensor: switch to using `qdata` ([https://github.com/pytorch/ao/pull/2787](https://github.com/pytorch/ao/pull/2787)) 
 * Nvfp4 tensor: switch to TorchAOBaseTensor ([https://github.com/pytorch/ao/pull/2788](https://github.com/pytorch/ao/pull/2788)) 
 * Nvfp4 tensor: refactor weight-only vs dynamic quant ([https://github.com/pytorch/ao/pull/2790](https://github.com/pytorch/ao/pull/2790)) 
 * Mxtensor: make data argument first and rename to `qdata` ([https://github.com/pytorch/ao/pull/2804](https://github.com/pytorch/ao/pull/2804)) 
 * Mxtensor: inherit from TorchAOBaseTensor ([https://github.com/pytorch/ao/pull/2805](https://github.com/pytorch/ao/pull/2805)) 
 * Mxtensor: refactor activation quant to use direct logic ([https://github.com/pytorch/ao/pull/2806](https://github.com/pytorch/ao/pull/2806)) 
 * Support more ops in TorchAOBaseTensor ([https://github.com/pytorch/ao/pull/2609](https://github.com/pytorch/ao/pull/2609))

## **New Contributors**

* @wdvr made their first contribution in [https://github.com/pytorch/ao/pull/2548](https://github.com/pytorch/ao/pull/2548) 
* @carmocca made their first contribution in [https://github.com/pytorch/ao/pull/2539](https://github.com/pytorch/ao/pull/2539) 
* @gausah-arm made their first contribution in [https://github.com/pytorch/ao/pull/2570](https://github.com/pytorch/ao/pull/2570) 
* @daniil-lyakhov made their first contribution in [https://github.com/pytorch/ao/pull/2581](https://github.com/pytorch/ao/pull/2581) 
* @zeshengzong made their first contribution in [https://github.com/pytorch/ao/pull/2599](https://github.com/pytorch/ao/pull/2599) 
* @amdfaa made their first contribution in [https://github.com/pytorch/ao/pull/2662](https://github.com/pytorch/ao/pull/2662) 
* @chowarfb made their first contribution in [https://github.com/pytorch/ao/pull/2657](https://github.com/pytorch/ao/pull/2657) 
* @abeakkas made their first contribution in [https://github.com/pytorch/ao/pull/2716](https://github.com/pytorch/ao/pull/2716) 
* @subhankarpal made their first contribution in [https://github.com/pytorch/ao/pull/2795](https://github.com/pytorch/ao/pull/2795)

**Full Changelog**: [https://github.com/pytorch/ao/compare/v0.12.0...v0.13.0-rc1](https://github.com/pytorch/ao/compare/v0.12.0...v0.13.0-rc1)