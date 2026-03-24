# pytorch/ao v0.12.0

Source: https://github.com/pytorch/ao/releases/tag/v0.12.0

*(Features, highlights, backwards-incompatible changes, deprecations, and new features only.)*

## Highlights

We are excited to announce the 0.12.0 release of torchao\! This release adds support for QAT \+ Axolotl Integration and prototype MXFP/NVFP support on Blackwell GPUs\!

### QAT \+ Axolotl Integration

TorchAO’s QAT support has been integrated into Axolotl’s fine-tuning recipes\! Check out the docs [here](https://docs.axolotl.ai/docs/qat.html) or run it yourself using the following command:

```shell
axolotl train examples/llama-3/3b-qat-fsdp2.yaml
axolotl quantize examples/llama-3/3b-qat-fsdp2.yaml
```

Initial results for Llama3.2-3B by @SalmanMohammadi ([https://github.com/axolotl-ai-cloud/axolotl/pull/2590](https://github.com/axolotl-ai-cloud/axolotl/pull/2590)):
| Model/Metric | hellaswag acc | hellaswag acc_norm | wikitext bits_per_byte | wikitext byte_perplexity | wikitext word_perplexity |
|--------------|---------------|-------------------|----------------------|-------------------------|-------------------------|
| bfloat16 | 0.5552 | 0.7315 | 0.6410 | 1.5594 | 10.7591 |
| bfloat16 PTQ | 0.5393 | 0.7157 | 0.6613 | 1.5815 | 11.6033 |
| qat ptq | 0.5423 | 0.7180 | 0.6567 | 1.5764 | 11.4043 |
| Recovered (qat ptq) | 18.87% | 14.56% | 22.66% | 23.08% | 23.57% |


### \[Prototype | API not finalized\] MXFP and NVFP support on Blackwell GPUs

TorchAO now includes prototype support for NVFP4 (NVIDIA's 4-bit floating-point format) and Microscaling (MX) formats on NVIDIA's latest Blackwell GPU architecture. These formats enable efficient inference, achieving up to 61% end-to-end performance improvement in vLLM on Qwen3 models and near 2x speedups for diffusion workloads.

To use:

```py
from torchao.quantization import quantize_ 
from torchao.prototype.mx_formats import (
 MXFPInferenceConfig,
 NVFP4InferenceConfig,
)
# Quantize model with MXFP8 
model = quantize_(model, MXFPInferenceConfig(block_size=32))
# Quantize model to NVFP4 (without double scaling)
model = quantize_(model, NVFP4InferenceConfig())
```

**Note**: This is a prototype feature with APIs subject to change. Requires NVIDIA Blackwell GPUs (B200, 5090\) with CUDA 12.8+.

---

## BC Breaking

* Remove preserve\_zero and zero\_point\_domain from choose\_qparams\_affine ([https://github.com/pytorch/ao/pull/2149](https://github.com/pytorch/ao/pull/2149)) 
* Rename qparams for tinygemm ([https://github.com/pytorch/ao/pull/2344](https://github.com/pytorch/ao/pull/2344)) 
* Convert quant\_primitives methods private ([https://github.com/pytorch/ao/pull/2350](https://github.com/pytorch/ao/pull/2350)) 
* Delete Galore ([https://github.com/pytorch/ao/pull/2397](https://github.com/pytorch/ao/pull/2397)) 
* Remove more Galore bits ([https://github.com/pytorch/ao/pull/2417](https://github.com/pytorch/ao/pull/2417)) 
* Remove `sparsity/prototype/blocksparse` ([https://github.com/pytorch/ao/pull/2205](https://github.com/pytorch/ao/pull/2205))

---

## Deprecations

* Clean up prototype folder ([https://github.com/pytorch/ao/pull/2232](https://github.com/pytorch/ao/pull/2232)) 
* Make float8 training's force\_recompute\_fp8\_weight\_in\_bwd flag do nothing ([https://github.com/pytorch/ao/pull/2356](https://github.com/pytorch/ao/pull/2356))

---

## New Features

* Enabling MOE Quantization using linear decomposition ([https://github.com/pytorch/ao/pull/2043](https://github.com/pytorch/ao/pull/2043)) 
* \[PT2E\]\[X86\] Migrate fusion passes in Inductor to torchao ([https://github.com/pytorch/ao/pull/2140](https://github.com/pytorch/ao/pull/2140)) 
* 2:4 activation sparsity packing kernels ([https://github.com/pytorch/ao/pull/2012](https://github.com/pytorch/ao/pull/2012)) 
* Add subclass based method for inference w/ MXFP8 ([https://github.com/pytorch/ao/pull/2132](https://github.com/pytorch/ao/pull/2132)) 
* Feat: Implementation of the DeepSeek blockwise quantization for fp8 tensors ([https://github.com/pytorch/ao/pull/1763](https://github.com/pytorch/ao/pull/1763)) 
* Arm\_inductor\_quantizer for Pt2e quantization ([https://github.com/pytorch/ao/pull/2139](https://github.com/pytorch/ao/pull/2139)) 
* Add mx\_fp4 path ([https://github.com/pytorch/ao/pull/2201](https://github.com/pytorch/ao/pull/2201)) 
* Add support for KleidiAI int4 kernels on aarch64 Linux ([https://github.com/pytorch/ao/pull/2169](https://github.com/pytorch/ao/pull/2169)) 
* Add support for fbgemm int4 mm kernel ([https://github.com/pytorch/ao/pull/2255](https://github.com/pytorch/ao/pull/2255)) 
* Enable fp16+int4 mixed precission path for int4 xpu path with int zero point ([https://github.com/pytorch/ao/pull/2240](https://github.com/pytorch/ao/pull/2240)) 
* Enable range learning for QAT ([https://github.com/pytorch/ao/pull/2033](https://github.com/pytorch/ao/pull/2033)) 
* Patch the \_is\_conv\_node function ([https://github.com/pytorch/ao/pull/2257](https://github.com/pytorch/ao/pull/2257)) 
* Add support for fbgemm fp8 kernels ([https://github.com/pytorch/ao/pull/2276](https://github.com/pytorch/ao/pull/2276)) 
* Add Float8ActInt4WeightQATQuantizer ([https://github.com/pytorch/ao/pull/2289](https://github.com/pytorch/ao/pull/2289)) 
* \[float8\] add \_auto\_filter\_for\_recipe to float8 ([https://github.com/pytorch/ao/pull/2410](https://github.com/pytorch/ao/pull/2410)) 
* NVfp4 ([https://github.com/pytorch/ao/pull/2408](https://github.com/pytorch/ao/pull/2408)) 
* \[float8\] Prevent quantize\_affine\_float8/dequantize\_affine\_float8 decomposed on inductor ([https://github.com/pytorch/ao/pull/2379](https://github.com/pytorch/ao/pull/2379)) 
* \[CPU\] Enable DA8W4 on CPU ([https://github.com/pytorch/ao/pull/2128](https://github.com/pytorch/ao/pull/2128)) 
* Add exportable coreml codebook quantization op ([https://github.com/pytorch/ao/pull/2443](https://github.com/pytorch/ao/pull/2443)) 
* Add support for Int4GroupwisePreshuffleTensor for fbgemm ([https://github.com/pytorch/ao/pull/2421](https://github.com/pytorch/ao/pull/2421))