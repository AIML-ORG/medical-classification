# pytorch/ao v0.15.0

Source: https://github.com/pytorch/ao/releases/tag/v0.15.0

*(Features, highlights, backwards-incompatible changes, deprecations, and new features only.)*

## Highlights

We are excited to announce the 0.15.0 release of torchao! This release adds:

- MXFP8 MoE training demonstrates 1.2x e2e training speedup with identical convergence versus bf16, training Llama4 Scout on a 64 node GB200 Crusoe cluster!
- MXFP8 MoE kernels shipped with torchao builds for CUDA 12.8+ (just pip install instead of building from source to use!)
- Safetensors enablement
- Quantization with parameter level targeting

### MXFP8 MoE training demonstrates 1.2x e2e training speedup with identical convergence versus bf16, training Llama4 Scout on a 64 node GB200 Crusoe cluster

Training runs on 64 node GB200 cluster with TorchTitan Llama4 Scout demonstrated a 1.2x e2e training speedup with equivalent convergence to bfloat16 training baseline. In fact, after 3,000 steps it finishes with slightly *lower* loss than bfloat16! This is consistent with our scaling experiments with [MXFP8 training for dense models](https://pytorch.org/blog/accelerating-2k-scale-pre-training-up-to-1-28x-with-torchao-mxfp8-and-torchtitan-on-crusoe-b200-cluster/).

 


| Number of GPUs | BF16 tokens/sec | MXFP8 tokens/sec | MXFP8 speedup vs BF16
| ----------------------- | --------------: | ----------------: | ---------------------: |
| 512 | 6169 | 7401 | 1.20x

See the [TorchAO MXFP8 MoE training documentation](https://github.com/pytorch/ao/blob/main/torchao/prototype/moe_training/README.md) for more details. You can also check out the [TorchTitan MXFP8 documentation](https://github.com/pytorch/torchtitan/blob/main/docs/mxfp8.md) to run pretraining jobs with TorchAO MXFP8 by adding a single config.
### Safetensors Enablement

You can now save and load TorchAO model checkpoints using safetensors! This feature is integrated with Hugging Face transformers starting from v5.0.0 and vLLM 0.13.0 for model inference/serving. 

We currently support the following stable configs:
`Float8DynamicActivationFloat8WeightConfig`
`Int4WeightOnlyConfig`
`IntxWeightOnlyConfig`
`Int8DynamicActivationIntxWeightConfig`
`Int8WeightOnlyConfig`
`Int8DynamicActivationInt8WeightConfig`


and will continue to add support for configs as they become stable in the future. 


Example:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from torchao.quantization import Float8WeightOnlyConfig


model_id = "facebook/opt-125m"
quant_config = Float8WeightOnlyConfig()
quantization_config = TorchAoConfig(quant_type=quant_config)
quantized_model = AutoModelForCausalLM.from_pretrained(
 model_id,
 device_map="auto",
 torch_dtype=torch.bfloat16,
 quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)


#### Push to hub
MODEL_NAME = model_id.split("/")[-1]
save_to = f"torchao-testing/{MODEL_NAME}-Float8WeightOnlyConfig-v2-0.15.0.dev-safetensors"
quantized_model.push_to_hub(save_to, safe_serialization=True)
tokenizer.push_to_hub(save_to)
```
To serve a safetensors model checkpoint on vLLM, use the CLI command `vllm serve torchao-testing/Qwen3-8B-INT4-0.15.0dev-safetensors`. 


### int4 weight only quantization with preshuffled packing format is supported in vllm ([#3245](https://github.com/pytorch/ao/pull/3245), [#26066](https://github.com/vllm-project/vllm/pull/26066))

Now we support running int4 weight only with preshuffled packing format in vllm. We can ship int4 weight only quantized checkpoint with plain packing format (e.g. https://huggingface.co/torchao-testing/opt-125m-Int4WeightOnlyConfig-v2-0.14.0.dev), and the Tensor with plain format will be converted to preshuffled format automatically in SM90+ and when fbgemm_gpu_genai is available.

### Support for quantizing any parameter by its fully qualified name (FQN) via `FqnToConfig` (#[3083](https://github.com/pytorch/ao/pull/3083))
```python
import torch
from torchao.quantization import quantize_, Float8WeightOnlyConfig, FqnToConfig


class Example(torch.nn.Module):
 def __init__(self):
 super().__init__()
 self.custom_name = torch.nn.Parameter(torch.Tensor([1]))


model = Example()

config = FqnToConfig({"custom_name": Float8WeightOnlyConfig()})

# filter_fn is disabled when quantizing by fqn
quantize_(model, config, filter_fn=None)
print(model.custom_name)
#Float8Tensor(self.act_quant_kwargs=None, self.qdata=tensor([448.], dtype=torch.float8_e4m3fn), self.scale=tensor([0.0022]), self.block_size=[1], self.mm_config=None, self.kernel_preference= self.shape=torch.Size([1]), self.device=device(type='cpu'), self.dtype=torch.float32)
```

This is to better support MoE models (Llama4, Deepseek, gpt-oss) which store their weights under attributes like "down_proj", "gate_up_proj", etc. Previously we assumed that the weights were always stored under "weights" as we were primarily focused on quantizing `nn.Linear` layers.

Currently parameter quantization is only enabled for `Int4WeightOnlyConfig`, `Float8DynamicActivationFloat8WeightConfig`, and `Float8WeightConfig`.

---

## BC Breaking

* Remove config functions like `int4_weight_only` (https://github.com/pytorch/ao/pull/3145)

Before:
```
from torchao.quantization import (
 float8_dynamic_activation_float8_weight,
 float8_static_activation_float8_weight,
 float8_weight_only,
 fpx_weight_only,
 gemlite_uintx_weight_only,
 int4_dynamic_activation_int4_weight,
 int4_weight_only,
 int8_dynamic_activation_int4_weight,
 int8_dynamic_activation_int8_weight,
 int8_weight_only,
 quantize_,
 uintx_weight_only,
)

quantize_(model, float8_dynamic_activation_float8_weight())
quantize_(model, float8_static_activation_float8_weight(torch.randn(3)))
quantize_(model, float8_weight_only())
quantize_(model, fpx_weight_only(3, 2))
quantize_(model, gemlite_uintx_weight_only())
quantize_(model, int4_dynamic_activation_int4_weight())
quantize_(model, int4_weight_only())
quantize_(model, int8_dynamic_activation_int4_weight())
quantize_(model, int8_dynamic_activation_int8_weight())
quantize_(model, int8_weight_only())
quantize_(model, uintx_weight_only(torch.uint4))
```

After:
```
from torchao.quantization import (
 Float8DynamicActivationFloat8WeightConfig,
 Float8StaticActivationFloat8WeightConfig,
 Float8WeightOnlyConfig,
 FPXWeightOnlyConfig,
 GemliteUIntXWeightOnlyConfig,
 Int4DynamicActivationInt4WeightConfig,
 Int4WeightOnlyConfig,
 Int8DynamicActivationInt4WeightConfig,
 Int8DynamicActivationInt8WeightConfig,
 Int8WeightOnlyConfig,
 quantize_,
 UIntXWeightOnlyConfig,
)

quantize_(model, Float8DynamicActivationFloat8WeightConfig())
quantize_(model, Float8StaticActivationFloat8WeightConfig(torch.randn(3)))
quantize_(model, Float8WeightOnlyConfig())
quantize_(model, FPXWeightOnlyConfig(3, 2))
quantize_(model, GemliteUIntXWeightOnlyConfig())
quantize_(model, Int4DynamicActivationInt4WeightConfig())
quantize_(model, Int4WeightOnlyConfig())
quantize_(model, Int8DynamicActivationInt4WeightConfig())
quantize_(model, Int8DynamicActivationInt8WeightConfig())
quantize_(model, Int8WeightOnlyConfig())
quantize_(model, UIntXWeightOnlyConfig(torch.uint4))
```

* Add quantize_ nn.Parameter support (https://github.com/pytorch/ao/pull/3083)

Before:
```
model = torch.nn.Sequential(
 torch.nn.Linear(128, 128), 
 torch.nn.Linear(128, 128), 
 torch.nn.Conv2d(128, 128, 3, 1, 1), 
).cuda().to(torch.bfloat16)

config = ModuleFqnToConfig({
 "0": Float8DynamicActivationFloat8WeightConfig(), 
})

# these are equivalent
quantize_(model, config, filter_fn=_is_linear)
quantize_(model, config, filter_fn=None)
quantize_(model, config)
```
After:

```
# VALID: user must specify None
quantize_(model, config, filter_fn=None)

# INVALID: these now error!
quantize_(model, config, filter_fn=_is_linear)
quantize_(model, config)
```

* Remove old TORCH_VERSION variables (https://github.com/pytorch/ao/pull/3146)

Before:
```
from torchao.utils import TORCH_VERSION_AT_LEAST_2_8
if TORCH_VERSION_AT_LEAST_2_8:
 print("PyTorch version was 2.8+")
```

After: (no replacement)

* 4/x: mx cleanup: use kernel_preference instead of gemm_kernel_choice (https://github.com/pytorch/ao/pull/3385)

---

## Deprecations

* Add deprecation warnings for various inference configs (https://github.com/pytorch/ao/pull/3294)
* 2/x mx cleanup: remove pack_fp6 (https://github.com/pytorch/ao/pull/3382)

---

## New Features

* Add learnable_fake_quantize in pt2e (https://github.com/pytorch/ao/pull/3135)
* Add learnablefakequantize to pt2e flow (https://github.com/pytorch/ao/pull/3170)
* Add per tensor fp8 quantization support for conv3d (https://github.com/pytorch/ao/pull/3215)
* Add a_1_128_w_128_128 (DeepSeek) float8 scaling for inference (https://github.com/pytorch/ao/pull/3257)
* Support int8 output for scaled_embedding_bag (https://github.com/pytorch/ao/pull/3231)
* Add per tensor fp8 conv2d support (https://github.com/pytorch/ao/pull/3315)
* add Float8OpaqueTensor for dynamic float8 act float8 weight (https://github.com/pytorch/ao/pull/3075)
* Add NPU (Ascend) backend support for INT4 weight-only quantization workflow (https://github.com/pytorch/ao/pull/3172)
* Add conv2d support for IntxUnpackedToInt8Tensor (https://github.com/pytorch/ao/pull/3371)
* Introduce SINQ calibration-free quantization algorithm (https://github.com/pytorch/ao/pull/3156)
* Add 2.9.1 compatibility (https://github.com/pytorch/ao/pull/3390)