# pytorch/ao v0.11.0

Source: https://github.com/pytorch/ao/releases/tag/v0.11.0

*(Features, highlights, backwards-incompatible changes, deprecations, and new features only.)*

## Highlights

We are excited to announce the 0.11.0 release of torchao\! This release adds support for mixture-of-experts (MoE) quantization, PyTorch 2 Export Quantization (PT2E), and a microbenchmarking framework for inference APIs\!

### MoE Quantization

We’ve a prototype feature for quantizing MoE modules with a number of TorchAO quantization techniques. This approach leverages the existing TorchAO features for quantizing linear ops and allows them to be used to quantize MoE modules.

```py
from torchao.quantization.prototype.moe_quant.utils import cond_ffn_filter, MoEQuantConfig
from torchao.quantization.quant_api import quantize_, Int8WeightOnlyConfig

quantize_(
 model, 
 MoEQuantConfig(Int8WeightOnlyConfig()), 
 filter_fn=cond_ffn_filter
)
model=torch.compile(
 model, 
 mode="reduce-overhead", 
 fullgraph=is_single_token_inference
)
```

While the above API is all that is needed to quantize a moe module if your moe module is written to be both quantizable and compilable, in practice its rare for a user model to satisfy these conditions due to the variety of MoE implementations. An initial swap of the normal MoE module with a `MoEFeedForwardAOQuantizable` module is needed to first prepare the model for quantization. An example of this can be found in `llama4_quant.py` where this technique is demonstrated for the huggingface llama-4-Scout-17B-16E-Instruct model.

We implemented MoE quantization with 2 methods. The first method (designated \`base\` in the below benchmarks) simply enhances the existing quantized tensor subclass to quantize the 3D MoE expert tensors and perform the necessary indexing and slicing ops while the second method (\`fake\`), uses a new tensor subclass to simulate a 3D quantized parameter by storing a sequence of 2D slices of the quantized parameter. The first approach is faster with marginally worse memory characteristics. In both cases doing MoE quantization in this way isn’t expected to be maximally performant compared to implementing fused MoE kernels for each technique, but this approach can yield both moderate speedups and significant memory savings.

The following benchmarks are for mixtral-moe run on a single H100 GPU:

| | batchsize 1 | | batchsize 8 | | | 
|-------------|-------------|-------------|-------------|--------------|-------------| 
| Technique | tok/s | memory (GB) | tok/s | tok/s\* batch | memory (GB) | 
| None | 78.35 | 93.76 | 18.2 | 145.64 | 94.12 | 
| int8wo-base | 98.4 | 48.87 | 4.94 | 39.56 | 49.2 | 
| int4wo-base | 79.38 | 36.15 | 10.29 | 82.29 | 36.12 | 
| fp8wo-base | 59.41 | 52.07 | 2.98 | 23.81 | 52.05 | 
| fp8dq-base | 45.92 | 53.97 | 3.78 | 30.23 | 53.94 | 
| int8wo-fake | 6.14 | 49.13 | 5.01 | 40.09 | 49.23 | 
| int4wo-fake | 14.25 | 30.21 | 11.84 | 94.75 | 30.19 | 
| fp8wo-fake | 3.2 | 50.31 | 2.88 | 23.08 | 50.29 | 
| fp8dq-fake | 9.78 | 50.92 | 4.08 | 32.61 | 50.89 |

### PT2 Export Quantization

We added pytorch 2 export quantization from pytorch to torchao. As part of the [planned migration](https://dev-discuss.pytorch.org/t/torch-ao-quantization-migration-plan/2810). We’ll follow up with adding deprecation warnings to PyTorch `torch.ao.quantization` APIs and updating docs in the future. We also simplified the import path for some of the util functions. Here is a non-exhaustive list of APIs you can use:

```
# top level APIs
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, prepare_qat_pt2e, convert_pt2e
from torchao.quantization.pt2e.quantizer import X86InductorQuantizer

# export utils
from torchao.quantization.pt2e import (
 move_exported_model_to_eval,
 move_exported_model_to_train,
 allow_exported_model_train_eval
)

# graph utils
from torchao.quantization.pt2e import (
 find_sequential_partitions,
 get_equivalent_types,
 update_equivalent_types_dict,
 bfs_trace_with_node_process,
)

 # pt2e numeric debugger
from torchao.quantization.pt2e import (
 generate_numeric_debug_handle,
 CUSTOM_KEY,
 NUMERIC_DEBUG_HANDLE_KEY,
 prepare_for_propagation_comparison,
 extract_results_from_loggers,
 compare_results,
)

```

### Microbenchmarking Framework for Inference APIs

We’ve introduced a streamlined [microbenchmark framework](https://github.com/pytorch/ao/blob/main/benchmarks/microbenchmarks/README.md), to help developers track and evaluate the performance of their post-training quantization and sparsity APIs for different matrix sizes and model types. The framework also includes support for advanced GPU and memory profiling techniques, providing deeper insights into performance characteristics.

To run the benchmarks, use the following command:

```
python -m benchmarks.microbenchmarks.benchmark_runner --config benchmarks/microbenchmarks/test/benchmark_config.yml
```

**Sample Benchmark Results (on 1xH100):**

| Name | Quantization | Shape | Baseline Inference Time (ms) | Inference Time (ms) | Speedup | 
|-------------------|-----------------|---------------------|------------------------------|---------------------|---------| 
| small\_bf16\_linear | float8dq-tensor | 16384, 16384, 16384 | 13.34 | 7.72 | 1.73x | 
| small\_bf16\_linear | float8dq-tensor | 16384, 16384, 32768 | 26.04 | 14.62 | 1.78x | 
| small\_bf16\_linear | float8dq-tensor | 16384, 16384, 65536 | 53.59 | 29.05 | 1.84x | 
| small\_bf16\_linear | float8dq-tensor | 16384, 32768, 32768 | 68.94 | 28.07 | 2.46x | 
| small\_bf16\_linear | float8dq-tensor | 16384, 32768, 65536 | 108.63 | 58.7 | 1.85x | 
| small\_bf16\_linear | float8dq-tensor | 16384, 65536, 65536 | 215.66 | 118.42 | 1.82x | 
| small\_bf16\_linear | float8dq-tensor | 32768, 32768, 32768 | 108.16 | 57.09 | 1.89x | 
| small\_bf16\_linear | float8dq-tensor | 32768, 32768, 65536 | 214.74 | 110.08 | 1.95x | 
| small\_bf16\_linear | float8dq-tensor | 32768, 65536, 65536 | 432.44 | 223.46 | 1.94x | 
| small\_bf16\_linear | float8dq-tensor | 65536, 65536, 65536 | 870.37 | 447.97 | 1.94x |

---

## BC Breaking

* Remove prototype low bit optim code completely ([https://github.com/pytorch/ao/pull/2159](https://github.com/pytorch/ao/pull/2159))

---

## New Features

* Add quantized attn\_scores @ v test for intented used in quantized attention ([https://github.com/pytorch/ao/pull/2008](https://github.com/pytorch/ao/pull/2008)) 
* Add fallback kernel and interface ([https://github.com/pytorch/ao/pull/2010](https://github.com/pytorch/ao/pull/2010)) 
* Add fallback kernel and interface for rhs only quantized matmul ([https://github.com/pytorch/ao/pull/2011](https://github.com/pytorch/ao/pull/2011)) 
* Add KleidiAI gemm kernels ([https://github.com/pytorch/ao/pull/2000](https://github.com/pytorch/ao/pull/2000)) 
* Use quantized gemm only on aarch64 ([https://github.com/pytorch/ao/pull/2023](https://github.com/pytorch/ao/pull/2023)) 
* Adds utility to replace Q/DQ ops with torchao quantized linear ops ([https://github.com/pytorch/ao/pull/1967](https://github.com/pytorch/ao/pull/1967)) 
* Adds Q/DQ layout support for embedding quantization with IntxWeightOnlyConfig ([https://github.com/pytorch/ao/pull/1972](https://github.com/pytorch/ao/pull/1972)) 
* Move Int8DynamicActivationIntxWeightConfig out of experimental ([https://github.com/pytorch/ao/pull/1968](https://github.com/pytorch/ao/pull/1968)) 
* Initial ParetoQ commit ([https://github.com/pytorch/ao/pull/1876](https://github.com/pytorch/ao/pull/1876)) 
* INT4 XPU enabling ([https://github.com/pytorch/ao/pull/1577](https://github.com/pytorch/ao/pull/1577)) 
* Vectorized row sum ([https://github.com/pytorch/ao/pull/2034](https://github.com/pytorch/ao/pull/2034)) 
* Add gemm for fp32\_a\_int8\_b matmul kernel ([https://github.com/pytorch/ao/pull/2039](https://github.com/pytorch/ao/pull/2039)) 
* Add gemm kernel to interface ([https://github.com/pytorch/ao/pull/2040](https://github.com/pytorch/ao/pull/2040)) 
* Add tests for attention matmul for gemm kernels ([https://github.com/pytorch/ao/pull/2041](https://github.com/pytorch/ao/pull/2041)) 
* Gemm int8 a int8 b kernels ([https://github.com/pytorch/ao/pull/2049](https://github.com/pytorch/ao/pull/2049)) 
* Add tests cases for q @ k attention variant ([https://github.com/pytorch/ao/pull/2051](https://github.com/pytorch/ao/pull/2051)) 
* Add gemm int8 a x int8 b to interface ([https://github.com/pytorch/ao/pull/2055](https://github.com/pytorch/ao/pull/2055)) 
* \[Quant\]\[PT2E\]\[X86\] Enable annotation of aten.mul.tensor with X86InductorQuantizer ([https://github.com/pytorch/ao/pull/2075](https://github.com/pytorch/ao/pull/2075)) 
* Add AOPerModuleConfig to `torchao.quantization` ([https://github.com/pytorch/ao/pull/2134](https://github.com/pytorch/ao/pull/2134)) 
* Enabling MoE Quantization using linear decomposition ([https://github.com/pytorch/ao/pull/2043](https://github.com/pytorch/ao/pull/2043))