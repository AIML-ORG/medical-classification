# pytorch/ao v0.14.0

Source: https://github.com/pytorch/ao/releases/tag/v0.14.0

**Note:** No `v0.14.0` release on GitHub; body is from [v0.14.1](https://github.com/pytorch/ao/releases/tag/v0.14.1).

*(Features, highlights, backwards-incompatible changes, deprecations, and new features only.)*

## BC Breaking

### Rename Int4WeightPreshuffledFakeQuantizeConfig ([https://github.com/pytorch/ao/pull/3005](https://github.com/pytorch/ao/pull/3005))

```py
from torchao.quantization import quantize_
from torchao.quantization.qat import (
 QATConfig,
 Int4WeightPreshuffledFakeQuantizeConfig,
)

# Before
fq_config = Int4WeightPreshuffledFakeQuantizeConfig()
quantize_(m, QATConfig(weight_config=fq_config)

# After
fq_config = Int4WeightFakeQuantizeConfig()
quantize_(m, QATConfig(weight_config=fq_config)
```

---

## Deprecations

### Bump `Int4WeightOnlyConfig` version from 1 to 2 ([https://github.com/pytorch/ao/pull/2949](https://github.com/pytorch/ao/pull/2949))

We updated the implementation for int4 Tensor, so bumps the default version from 1 to 2 for these two configs.

```
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "torchao-testing/opt-125m-Int4WeightOnlyConfig-v1-0.14.dev"
quantized_model = AutoModelForCausalLM.from_pretrained(
 model_name,
 torch_dtype="bfloat16",
 device_map="cuda",
)

/data/users/jerryzh/ao/torchao/core/config.py:250: UserWarning: Stored version is not the same as current default version of the config: stored_version=1, current_default_version=2, please check the deprecation warning
 warnings.warn(
/data/users/jerryzh/ao/torchao/dtypes/uintx/tensor_core_tiled_layout.py:241: UserWarning: Models quantized with version 1 of Int4WeightOnlyConfig is deprecated and will no longer be supported in a future release, please upgrade torchao and quantize again, or download a newer torchao checkpoint, see https://github.com/pytorch/ao/issues/2948 for more details
 warnings.warn(
```

Suggestion: upgrade torchao to 0.14 and later and generate the checkpoint again:

```
quantize_(model, Int4WeightOnlyConfig(group_size=128))
```

Or download the checkpoint again (please let us know if the checkpoint is not updated)

Please see [\#2948](https://github.com/pytorch/ao/issues/2948) for more details around the deprecation.

### Deprecate config functions like `int4_weight_only` ([https://github.com/pytorch/ao/pull/2994](https://github.com/pytorch/ao/pull/2994))

The int4\_weight\_only functions have been superseded by `AOBaseConfig` objects like `Int4WeightOnlyConfig(...)`, which have been in-use since several previous releases

```py
from torchao.quantization import (
 Int4WeightOnlyConfig,
 int4_weight_only,
 quantize_,
)

# Before
quantize_(m, int4_weight_only())

# After
quantize_(m, Int4WeightOnlyConfig())

# Full list of deprecated functions
float8_dynamic_activation_float8_weight
float8_static_activation_float8_weight
float8_weight_only
fpx_weight_only
gemlite_uintx_weight_only
int4_dynamic_activation_int4_weight
int4_weight_only
int8_dynamic_activation_int4_weight
int8_dynamic_activation_int8_weight
int8_weight_only
uintx_weight_only
```

---

## New Features
* CPU 
 * Introduce Int4OpaqueTensor to replace Int4CPULayout in AQT ([https://github.com/pytorch/ao/pull/2798](https://github.com/pytorch/ao/pull/2798)) 
 * \[float8\] Add scaled\_embedding\_bag kernel ([https://github.com/pytorch/ao/pull/2686](https://github.com/pytorch/ao/pull/2686)) 
* Safetensor Support for TorchAO Configs: 
 * Add int4tensor support for safetensors ([https://github.com/pytorch/ao/pull/3056](https://github.com/pytorch/ao/pull/3056)) 
 * Add int4tilepackedto4dtensor subclass to safetensors ([https://github.com/pytorch/ao/pull/3064](https://github.com/pytorch/ao/pull/3064)) 
 * Add IntxUnpackedToInt8Tensor to safetensors ([https://github.com/pytorch/ao/pull/3065](https://github.com/pytorch/ao/pull/3065)) 
* Add Int4PlainInt32Tensor ([https://github.com/pytorch/ao/pull/2845](https://github.com/pytorch/ao/pull/2845)) 
* Add from\_int4\_tensor in Int4PreshuffledTensor ([https://github.com/pytorch/ao/pull/2978](https://github.com/pytorch/ao/pull/2978)) 
* Add hqq support for Int4TilePackedTo4dTensor ([https://github.com/pytorch/ao/pull/2912](https://github.com/pytorch/ao/pull/2912)) 
* Add torchao\_convert to PARQ's QuantOptimizer ([https://github.com/pytorch/ao/pull/2947](https://github.com/pytorch/ao/pull/2947)) 
* Add scale-only version of the HQQ algorithm for IntxWeightOnlyConfig/Int8DynamicActivationIntxWeightConfig ([https://github.com/pytorch/ao/pull/3110](https://github.com/pytorch/ao/pull/3110))