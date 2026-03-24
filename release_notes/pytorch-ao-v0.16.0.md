# pytorch/ao v0.16.0

Source: https://github.com/pytorch/ao/releases/tag/v0.16.0

*(Features, highlights, backwards-incompatible changes, deprecations, and new features only.)*

## Highlights

We are excited to announce the 0.16.0 release of torchao! This release adds support for MXFP8 MoE Building Blocks for Training with Expert Parallelism and deprecated older versions of some configs and less used quantization options to keep torchao leaner! We also revamped our [doc page](https://docs.pytorch.org/ao/main/), [README](https://github.com/pytorch/ao/blob/main/README.md) and made some progress in making torchao [ABI stable](https://github.com/pytorch/ao/issues/3516).

### MXFP8 MoE Building Blocks for Training with Expert Parallelism

This release includes the following differentiable building blocks for MXFP8 MoE Training with expert parallelism:

* [a2a\_dispatch\_mxfp8\_fwd\_hp\_bwd](https://github.com/pytorch/ao/blob/eb39a84e6ef89de70f0734877eaa521e0ac39d43/torchao/prototype/moe_training/ep/a2a_dispatch.py#L143): All-to-all token dispatch (MXFP8 forward pass, BF16 backward pass) 
* [permute\_mxfp8\_fwd\_hp\_bwd](https://github.com/pytorch/ao/blob/eb39a84e6ef89de70f0734877eaa521e0ac39d43/torchao/prototype/moe_training/ep/permute.py#L212): Permute and pad tokens for MXFP8 computation (MXFP8 forward pass, BF16 backward pass) 
* [\_to\_mxfp8\_then\_scaled\_grouped\_mm](https://github.com/pytorch/ao/blob/eb39a84e6ef89de70f0734877eaa521e0ac39d43/torchao/prototype/moe_training/scaled_grouped_mm.py#L1006): MXFP8 grouped GEMM for routed expert computation (**new:** optionally accepts pre-quantized inputs). Produces bfloat16 output. 
* [unpermute\_hp\_fwd\_mxfp8\_bwd](https://github.com/pytorch/ao/blob/eb39a84e6ef89de70f0734877eaa521e0ac39d43/torchao/prototype/moe_training/ep/unpermute.py#L115): Unpermute tokens back to original order (BF16 forward pass, MXFP8 backward pass) 
* [a2a\_combine\_hp\_fwd\_mxfp8\_bwd](https://github.com/pytorch/ao/blob/eb39a84e6ef89de70f0734877eaa521e0ac39d43/torchao/prototype/moe_training/ep/a2a_combine.py#L161): All-to-all token combine (BF16 forward pass, MXFP8 backward pass). Note the actual combine/aggregation op does not happen here, the naming is just to indicate it is intended to be used for the all2all immediatley preceding the aggregation.

These autograd functions can be chained together to implement efficient MoE training with expert parallel comms and grouped GEMMs in MXFP8.

This approach achieves 10% \- 25% tokens/second speedup for DeepSeekV3 16b training:

* \+10% tokens/second on single node 8xB200 with NVLink intra-node networking for inter-device communication. 
* \+25% tokens/second on multi-node B200 cluster with IB inter-node networking and NVLink intra-node networking.

---

## Deprecations

* Deprecate v1 of `Float8WeightOnlyConfig`, `Float8DynamicActivationFloat8WeightConfig`, `Int8DynamicActivationIntxWeightConfig`, `IntxWeightOnlyConfig`, `Int4WeightOnlyConfig` ([https://github.com/pytorch/ao/pull/3510](https://github.com/pytorch/ao/pull/3510), [https://github.com/pytorch/ao/pull/3511](https://github.com/pytorch/ao/pull/3511), [https://github.com/pytorch/ao/pull/3512](https://github.com/pytorch/ao/pull/3512), [https://github.com/pytorch/ao/pull/3513](https://github.com/pytorch/ao/pull/3513))

```py
# v0.15.0 - version 1 was available.
config = Float8WeightOnlyConfig(version=1, ...)
config = Float8DynamicActivationFloat8WeightConfig(version=1, ...)
config = Int8DynamicActivationIntxWeightConfig(version=1, ...)
config = IntxWeightOnlyConfig(version=1, ...)
config = Int4WeightOnlyConfig(version=1, ...)

# v0.16.0 - use version 2 (default). Using version 1 is no longer supported.
config = Float8WeightOnlyConfig(version=2, ...)
config = Float8DynamicActivationFloat8WeightConfig(version=2, ...)
config = Int8DynamicActivationIntxWeightConfig(version=2, ...)
config = IntxWeightOnlyConfig(version=2, ...)
config = Int4WeightOnlyConfig(version=2, ...)
```

* Move `Int8DynamicActivationInt4WeightConfig`, `Int4DynamicActivationInt4WeightConfig`, `GemliteUIntXWeightOnlyConfig`, `Float8StaticActivationFloat8WeightConfig`, `UIntXWeightOnlyConfig`, `FPXWeightOnlyConfig` to prototype ([https://github.com/pytorch/ao/pull/3491](https://github.com/pytorch/ao/pull/3491))

```py
# v0.15.0
from torchao.quantization import (
 Int8DynamicActivationInt4WeightConfig, 
 Int4DynamicActivationInt4WeightConfig, 
 GemliteUIntXWeightOnlyConfig, 
 Float8StaticActivationFloat8WeightConfig, 
 UIntXWeightOnlyConfig, 
 # removed in this release
 FPXWeightOnlyConfig,
)

# v0.16.0. These workflows may be deleted in a future version
from torchao.prototype.quantization.quant_api import (
 Int8DynamicActivationInt4WeightConfig, 
 GemliteUIntXWeightOnlyConfig, 
 Float8StaticActivationFloat8WeightConfig, 
 UIntXWeightOnlyConfig, 
 # removed in this release
 FPXWeightOnlyConfig,
 Int4DynamicActivationInt4WeightConfig,
)
```

* Remove `Int4DynamicActivationInt4WeightConfig`, `FPXWeightOnlyConfig`, `Float8DynamicActivationFloat8SemiSparseWeightConfig and SRELUFloat8SemiSparseDynamicActivationFLoat8WeightConfig` ([https://github.com/pytorch/ao/pull/3723](https://github.com/pytorch/ao/pull/3723), [https://github.com/pytorch/ao/pull/3520](https://github.com/pytorch/ao/pull/3520), [https://github.com/pytorch/ao/pull/3744](https://github.com/pytorch/ao/pull/3744))

```py
# v0.15.0
config = Int4DynamicActivationInt4WeightConfig()
config = FPXWeightOnlyConfig(3, 2)
config = fpx_weight_only(3, 2) # deprecated alias
config = Float8DynamicActivationFloat8SemiSparseWeightConfig() 
config = SRELUFloat8SemiSparseDynamicActivationFloat8WeightConfig() 
quantize_(model, config)

# v0.16.0
# 
The configs are dropped. Please use torchao <= 0.15.0 to use these configs.
```

* Remove `CutlassInt4PackedLayout`, `MarlinQQQLayout`, `CutlassSemiSparseLayout` ([https://github.com/pytorch/ao/pull/3723](https://github.com/pytorch/ao/pull/3723), [https://github.com/pytorch/ao/pull/3612](https://github.com/pytorch/ao/pull/3612), [https://github.com/pytorch/ao/pull/3613](https://github.com/pytorch/ao/pull/3613), [https://github.com/pytorch/ao/pull/3744](https://github.com/pytorch/ao/pull/3744))

```py
# v0.15.0
## CutlassInt4PackedLayout
config = Int8DynamicActivationInt4WeightConfig(
 group_size=None,
 mapping_type=MappingType.SYMMETRIC,
 act_mapping_type=MappingType.SYMMETRIC,
 layout=CutlassInt4PackedLayout(),
)

## MarlinQQQLayout
config = Int8DynamicActivationInt4WeightConfig(layout=MarlinQQQLayout())
config = Int4DynamicActivationInt4WeightConfig(layout=MarlinQQQLayout())

## MarlinSparseLayout
apply_fake_sparsity(model) 
config = Int4WeightOnlyConfig(layout=MarlinSparseLayout(), version=1) 
config = Int4WeightOnlyConfig(int4_packing_format="marlin_sparse", version=2) 

## CutlassSemiSparseLayout
config = Float8DynamicActivationFloat8SemiSparseWeightConfig(layout=CutlassSemiSparseLayout())

## quantizing the model with the config
quantize_(model, config)

# v0.16.0
# The config and layout options are dropped. Please use torchao <= 0.15.0 to use the layout
```

* Remove Old GPTQ Implementation ([https://github.com/pytorch/ao/pull/3720](https://github.com/pytorch/ao/pull/3720))

```py
# v0.15.0
from torchao.quantization import MultiTensorInputRecorder, Int4WeightOnlyGPTQQuantizer
model = get_model()
input_recorder = MultiTensorInputRecorder()
for i in range(calibration_limit):
 args = get_next_input()
 input_recorder(*args)
quantizer = Int4WeightOnlyGPTQQuantizer()
args = input_recorder.get_recorded_inputs()
quantizer.quantize(model, *args)
args = get_next_input()
out = model(*args)

# v0.16.0 - this functionality is deleted. We are starting over
# for GPTQ in https://github.com/pytorch/ao/tree/main/torchao/prototype/gptq
```

* Delete Old SmoothQuant Implementation ([https://github.com/pytorch/ao/pull/3495](https://github.com/pytorch/ao/pull/3495))

```py
# v0.15.0 from torchao.quantization.smoothquant import (
 swap_linear_with_smooth_fq_linear,
 smooth_fq_linear_to_inference,
)

swap_linear_with_smooth_fq_linear(model_copy, alpha=0.75)
model_copy(**encoded_input)
smooth_fq_linear_to_inference(model_copy)

# v0.16.0 - the functionality above is deleted. Use `torchao.prototype.smoothquant`
# instead (https://github.com/pytorch/ao/tree/main/torchao/prototype/smoothquant) 
```

* Add deprecation warning to `torchao.autoquant` ([https://github.com/pytorch/ao/pull/3741](https://github.com/pytorch/ao/pull/3741))

This functionality will be removed in a future release of torchao. Please see [https://github.com/pytorch/ao/issues/3739](https://github.com/pytorch/ao/issues/3739) for context.

---

## New Features

* Add INT8 Static Quantization Workflow ([https://github.com/pytorch/ao/pull/3442](https://github.com/pytorch/ao/pull/3442)) 
* \[Prototype\] Add support for MXFP8 and MXFP4 QAT ([https://github.com/pytorch/ao/pull/3644](https://github.com/pytorch/ao/pull/3644)) 
* \[Prototype\] MXFP8 MoE training 
 * Add `wgrad_with_hp` option for mxfp8 moe training ([https://github.com/pytorch/ao/pull/3508](https://github.com/pytorch/ao/pull/3508)) 
 * MXFP8 `a2a_dispatch` autograd function ([https://github.com/pytorch/ao/pull/3579](https://github.com/pytorch/ao/pull/3579)) 
 * MXFP8 token `permute` autograd func \+ triton kernels ([https://github.com/pytorch/ao/pull/3580](https://github.com/pytorch/ao/pull/3580)) 
 * MXFP8 `unpermute` autograd function ([https://github.com/pytorch/ao/pull/3581](https://github.com/pytorch/ao/pull/3581)) 
 * MXFP8 a2a\_combine autograd function ([https://github.com/pytorch/ao/pull/3582](https://github.com/pytorch/ao/pull/3582)) 
 * Handle pre-quantized inputs/grads in forward/backward of \_MXFP8GroupedMM autograd func ([https://github.com/pytorch/ao/pull/3583](https://github.com/pytorch/ao/pull/3583)) 
 * Add benchmark for e2e mxfp8 EP pipeline ([https://github.com/pytorch/ao/pull/3585](https://github.com/pytorch/ao/pull/3585)) 
 * Export `wgrad_with_hp` recipe to quantize\_ api ([https://github.com/pytorch/ao/pull/3611](https://github.com/pytorch/ao/pull/3611)) 
 * Fallback cuda kernel for when input doesn't meet 2d TMA constraints ([https://github.com/pytorch/ao/pull/3708](https://github.com/pytorch/ao/pull/3708)) 
 * Add `emulated` mode ([https://github.com/pytorch/ao/pull/3724](https://github.com/pytorch/ao/pull/3724)) 
 * Add support for MXFP8 All gather ([https://github.com/pytorch/ao/pull/3435](https://github.com/pytorch/ao/pull/3435)) 
 * Integrate cuda kernel for 'groups along M scale blocked layout' ([https://github.com/pytorch/ao/pull/3556](https://github.com/pytorch/ao/pull/3556)) 
 * Default to triton kernel for dim0 cast ([https://github.com/pytorch/ao/pull/3560](https://github.com/pytorch/ao/pull/3560)) 
 * `_to_mxfp8_then_scaled_grouped_mm` wrapper that accepts keyword args ([https://github.com/pytorch/ao/pull/3561](https://github.com/pytorch/ao/pull/3561)) 
 * auto-select chunk\_width in cuda blocked layout kernel ([https://github.com/pytorch/ao/pull/3658](https://github.com/pytorch/ao/pull/3658)) 
 * `scaled_grouped_mm` support gfx942 fp8 data type ([https://github.com/pytorch/ao/pull/3540](https://github.com/pytorch/ao/pull/3540)) 
 * Add custom sharding for triton dim0 quant kernel ([https://github.com/pytorch/ao/pull/3812](https://github.com/pytorch/ao/pull/3812)) 
 * Bug fix and test updates related to new triton\_calculate\_scale param ([https://github.com/pytorch/ao/pull/3522](https://github.com/pytorch/ao/pull/3522)) 
 * Update bench script to use new cuda wrapper ([https://github.com/pytorch/ao/pull/3562](https://github.com/pytorch/ao/pull/3562)) 
 * Fix torch ref impl of SF blocked layout per group along K ([https://github.com/pytorch/ao/pull/3603](https://github.com/pytorch/ao/pull/3603)) 
 * Only use `torch._dynamo.nonstrict_trace` if it exists in the torch version ([https://github.com/pytorch/ao/pull/3650](https://github.com/pytorch/ao/pull/3650)) 
 * cuda blocked layout kernel handling for skinnier scale tensors ([https://github.com/pytorch/ao/pull/3656](https://github.com/pytorch/ao/pull/3656)) 
 * Fix bench script bug for dim0 mxfp8 rceil ([https://github.com/pytorch/ao/pull/3665](https://github.com/pytorch/ao/pull/3665)) 
 * Register constant with pytree ([https://github.com/pytorch/ao/pull/3667](https://github.com/pytorch/ao/pull/3667)) 
 * Update tensor subclass emulated param name ([https://github.com/pytorch/ao/pull/3811](https://github.com/pytorch/ao/pull/3811))