# pytorch/pytorch v2.5.0

Source: https://github.com/pytorch/pytorch/releases/tag/v2.5.0

*(Features, highlights, backwards-incompatible changes, deprecations, and new features only.)*

## Highlights

We are excited to announce the release of PyTorch® 2.5! This release features a new CuDNN backend for SDPA, enabling speedups by default for users of SDPA on H100s or newer GPUs. As well, regional compilation of torch.compile offers a way to reduce the cold start up time for torch.compile by allowing users to compile a repeated nn.Module (e.g. a transformer layer in LLM) without recompilations. Finally, TorchInductor CPP backend offers solid performance speedup with numerous enhancements like FP16 support, CPP wrapper, AOT-Inductor mode, and max-autotune mode.
This release is composed of 4095 commits from 504 contributors since PyTorch 2.4. We want to sincerely thank our dedicated community for your contributions. As always, we encourage you to try these out and report any issues as we improve 2.5. More information about how to get started with the PyTorch 2-series can be found at our [Getting Started](https://pytorch.org/get-started/pytorch-2.0/) page.
As well, please check out our new ecosystem projects releases with [TorchRec](https://github.com/pytorch/torchrec) and [TorchFix](https://github.com/pytorch-labs/torchfix/releases/tag/v0.6.0).

| Beta | Prototype |
|------|-----------|
| CuDNN backend for SDPA | FlexAttention |
| torch.compile regional compilation without recompilations | Compiled Autograd |
| TorchDynamo added support for exception handling & MutableMapping types | Flight Recorder |
| TorchInductor CPU backend optimization | Max-autotune Support on CPU with GEMM Template |
| | TorchInductor on Windows |
| | FP16 support on CPU path for both eager mode and TorchInductor CPP backend |
| | Autoload Device Extension |
| | Enhanced Intel GPU support |

*To see a full list of public feature submissions click [here](https://docs.google.com/spreadsheets/d/1TzGkWuUMF1yTe88adz1dt2mzbIsZLd3PBasy588VWgk/edit?gid=949287277#gid=949287277).

### BETA FEATURES
#### [Beta] CuDNN backend for SDPA
The cuDNN "Fused Flash Attention" backend was landed for `torch.nn.functional.scaled_dot_product_attention`. On NVIDIA H100 GPUs this can provide up to 75% speed-up over FlashAttentionV2. This speedup is enabled by default for all users of SDPA on H100 or newer GPUs.
#### [Beta] _torch.compile_ regional compilation without recompilations
Regional compilation without recompilations, via `torch._dynamo.config.inline_inbuilt_nn_modules` which default to True in 2.5+. This option allows users to compile a repeated nn.Module (e.g. a transformer layer in LLM) without recompilations. Compared to compiling the full model, this option can result in smaller compilation latencies with 1%-5% performance degradation compared to full model compilation.

See the [tutorial](https://pytorch.org/tutorials/recipes/regional_compilation.html) for more information.
#### [Beta] TorchInductor CPU backend optimization
This feature advances Inductor’s CPU backend optimization, including CPP backend code generation and FX fusions with customized CPU kernels. The Inductor CPU backend supports vectorization of common data types and all Inductor IR operations, along with the static and symbolic shapes. It is compatible with both Linux and Windows OS and supports the default Python wrapper, the CPP wrapper, and AOT-Inductor mode. 

Additionally, it extends the max-autotune mode of the GEMM template (prototyped in 2.5), offering further performance gains. The backend supports various FX fusions, lowering to customized kernels such as oneDNN for Linear/Conv operations and SDPA. The Inductor CPU backend consistently achieves performance speedups across three benchmark suites—TorchBench, Hugging Face, and timms—outperforming eager mode in 97.5% of the 193 models tested.

### PROTOTYPE FEATURES
#### [Prototype] FlexAttention
We've introduced a flexible API that enables implementing various attention mechanisms such as Sliding Window, Causal Mask, and PrefixLM with just a few lines of idiomatic PyTorch code. This API leverages torch.compile to generate a fused FlashAttention kernel, which eliminates extra memory allocation and achieves performance comparable to handwritten implementations. Additionally, we automatically generate the backwards pass using PyTorch's autograd machinery. Furthermore, our API can take advantage of sparsity in the attention mask, resulting in significant improvements over standard attention implementations.

For more information and examples, please refer to the [official blog post](https://pytorch.org/blog/flexattention/) and [Attention Gym](https://github.com/pytorch-labs/attention-gym).
#### [Prototype] Compiled Autograd
Compiled Autograd is an extension to the PT2 stack allowing the capture of the entire backward pass. Unlike the backward graph traced by AOT dispatcher, Compiled Autograd tracing is deferred until backward execution time, which makes it impervious to forward pass graph breaks, and allows it to record backward hooks into the graph.

Please refer to the [tutorial](https://pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html) for more information.
#### [Prototype] Flight Recorder
Flight recorder is a new debugging tool that helps debug stuck jobs. The tool works by continuously capturing information about collectives as they run. Upon detecting a stuck job, the information can be used to quickly identify misbehaving ranks/machines along with code stack traces.

For more information please refer to the following [tutorial](https://pytorch.org/tutorials/prototype/flight_recorder_tutorial.html).
#### [Prototype] Max-autotune Support on CPU with GEMM Template
Max-autotune mode for the Inductor CPU backend in torch.compile profiles multiple implementations of operations at compile time and selects the best-performing one. This is particularly beneficial for GEMM-related operations, using a C++ template-based GEMM implementation as an alternative to the ATen-based approach with oneDNN and MKL libraries. We support FP32, BF16, FP16, and INT8 with epilogue fusions for x86 CPUs. We’ve seen up to 7% geomean speedup on the dynamo benchmark suites and up to 20% boost in next-token latency for LLM inference.

For more information please refer to the [tutorial](https://pytorch.org/tutorials/prototype/max_autotune_on_CPU_tutorial.html).
#### [Prototype] TorchInductor CPU on Windows
Inductor CPU backend in torch.compile now works on Windows. We support MSVC (cl), clang (clang-cl) and Intel compiler (icx-cl) for Windows inductor currently.

See the [tutorial](https://pytorch.org/tutorials/prototype/inductor_windows_cpu.html) for more details.
#### [Prototype] FP16 support on CPU path for both eager mode and TorchInductor CPP backend
Float16 is a commonly used reduced floating point type for performance improvement in neural network inference/training. Since this release, float16 for both eager and TorchInductor is supported on the CPU path.
#### [Prototype] Autoload Device Extension
PyTorch now supports autoloading for out-of-tree device extensions, streamlining integration by eliminating the need for manual imports. This feature, enabled through the torch.backends entrypoint, simplifies usage by ensuring seamless extension loading, while allowing users to disable it via an environment variable if needed.

See the [tutorial](https://pytorch.org/tutorials/prototype/python_extension_autoload.html) for more information.
#### [Prototype] Enhanced Intel GPU support
Intel GPUs support enhancement is now available for both Intel® Data Center GPU Max Series and Intel® Client GPUs (Intel® Core™ Ultra processors with built-in Intel® Arc™ graphics and Intel® Arc™ Graphics for dGPU parts), which is to make it easier to accelerate your Machine Learning workflows on Intel GPUs in PyTorch 2.5 release. We also enabled the initial support of PyTorch on Windows for Intel® Client GPUs in this release.
- Expanded PyTorch hardware backend support matrix to include both Intel Data Center and Client GPUs.   
- The implementation of SYCL* kernels to enhance coverage and execution of Aten operators on Intel GPUs to boost performance in PyTorch eager mode. 
- Enhanced Intel GPU backend of torch.compile to improve inference and training performance for a wide range of deep learning workloads. 

These features are available through PyTorch preview and nightly binary PIP wheels. For more information regarding Intel GPU support, please refer to [documentation](https://pytorch.org/docs/main/notes/get_start_xpu.html).

---

## Backwards Incompatible changes

### Distributed

- [c10d] Remove Option for ProcessGroup and Expose backend Options to reflect the correct code structure (#132931)
 - We released Dispatchable collectives in 2.0 and we will use Backend Option for Backend initialization and the PG options are not needed any more.
 - In 2.4 and before, users can do:
 ```py
 # Users can pass in a basic option when creating an instance of ProcessGroup
 base_pg_options = ProcessGroup.Options(backend=str(backend))
 base_pg_options._timeout = timeout

 pg: ProcessGroup = ProcessGroup(
 store, rank, group_size, base_pg_options
 )

 # Users then need to create a backend option to create the comm backend (e.g., ProcessGroupNCCL)
 pg_options = ProcessGroupNCCL.Options()
 backend = ProcessGroupNCCL(
 store, rank, group_size, pg_options
 )
 ```
 - But from 2.5 onwards, users don’t need to pass in an option to create an instance of ProcessGroup and user can still set default backend for the pg since users still try to get default backend in the code:

 ```py
 # No basic option is passed in when creating a instance of ProcessGroup
 pg: ProcessGroup = ProcessGroup(store, rank, group_size)
 pg._set_default_backend(Backend.backend_type_map[backend])
 # Users then need to create a backend option to create the comm backend (e.g., ProcessGroupNCCL)
 pg_options = ProcessGroupNCCL.Options()
 backend = ProcessGroupNCCL(
 store, rank, group_size, pg_options
 )
 ```

### Export

- Remove `dynamic_dim()` (#134211) 
 - The `dynamic_dim()` method for specifying dynamic shapes in `torch.export()` has been removed. Please refer to the [export tutorial](https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html#constraints-dynamic-shapes) for using `Dims` to specify dynamic shapes.

### Inductor

- [Torch] Support meta device in checkpoint (#132684)
- Switch to internal benchmarking and update benchmarking path (#132827)

 This change moves from using triton’s benchmarking utils to the internal inductor utils at `torch._inductor.runtime.benchmarking`. To update your benchmarking code:

 ```py
 # before
 from torch._inductor.runtime.runtime_utils import do_bench_gpu
 # ...
 do_bench_gpu(kernel, rep=40, fast_flush=True)

 # after
 from torch._inductor.runtime.benchmarking import benchmarker
 # ...
 benchmarker.benchmark_gpu(kernel_call, rep=40, fast_flush=True)
 ```

### mps

- [MPS][BE] Delete MacOS-12.3 specific checks (#133141)

### nn

- Update fused kernels and call _safe_softmax from SDPA (#131863) 
 
 Before this PR, fully masked rows in the `attn_mask` passed to `nn.functional.scaled_dot_product_attention` would yield NANs in the output, after this PR, fully masked rows yield 0s.

 Example:

 2.4.0

 ```py 
 B, S, D = 1, 1, 128

 q = torch.randn(B, S, D, device='cuda') 
 k = torch.randn(B, S, D, device='cuda') 
 v = torch.randn(B, S, D, device='cuda')

 attn_mask = torch.tensor([False], device='cuda')

 F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask) 
 tensor([[[nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 
 nan, nan, nan, nan, nan, nan, nan, nan, nan]]], device='cuda:0') 
 ```

 2.5.0

 ```py 
 B, S, D = 1, 1, 128

 q = torch.randn(B, S, D, device='cuda') 
 k = torch.randn(B, S, D, device='cuda') 
 v = torch.randn(B, S, D, device='cuda')

 attn_mask = torch.tensor([False], device='cuda')

 F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask) 
 tensor([[[-0., -0., 0., -0., 0., 0., -0., -0., -0., -0., 0., -0., -0., -0., 0., -0., 0., 0., -0., -0., 0., -0., -0., 
 0., 0., -0., 0., -0., -0., 0., -0., -0.]]], device='cuda:0') 
 ```

### Optimizer Frontend

- Add support to `GradScaler` for respecting an already set `grad_scale` value (#123429)

### Python Frontend

- No more CPython 3.8 support and removal from binary (#132138) 
 CPython 3.8 is now EOL and PyTorch 2.4 is the last version that is supported. 
 See [https://devguide.python.org/versions/](https://devguide.python.org/versions/) for CPython EOL timelines and [https://github.com/pytorch/pytorch/blob/main/RELEASE.md\#python](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#python) for PyTorch’s support CPython version policy.

### ONNX
#### Options to `torch.onnx.export` (except for the first three arguments) are now keyword-only (#131501)

Options can be supplied by keywords only to allow for future addition and evolution of the `torch.onnx.export` API.

Example: 
Version 2.4 
```python 
torch.onnx.export(model, input, f, True, False) 
```

Version 2.5: 
```python 
torch.onnx.export(model, input, f, export_params=True, verbose=False) 
```

#### Deprecated internal API `torch.onnx._export` has been removed (133824)

`torch.onnx._export` is an internal API which is not meant for public consumption. Use the public `torch.onnx.export` instead.

Example: 
Version 2.4 
```python 
torch.onnx._export(...) 
```

Version 2.5: 
```python 
torch.onnx.export(...) 
```

#### The `op_level_debug` option from `torch.onnx.ExportOptions` has been removed (#134961)

This option, designed to identify operator discrepancies, proved unreliable and has been removed. Instead, use `torch.onnx.export(..., report=True, verify=True)` option to validate exported models.

#### The `ONNXProgramSerializer` class has been removed (#135261)

The ONNX model in `torch.onnx.ONNXProgram` is now maintained and serialized by [ONNX IR](https://github.com/microsoft/onnxscript/blob/main/onnxscript/ir/README.md). 
`textproto`, `onnxtext`, and `json` formats are supported by default when calling `ONNXProgram.save()` with a corresponding file extension.

#### The `SymbolicContext` class has been removed (#132184)

The deprecated `torch.onnx.SymbolicContext` class has been removed. (Non-dynamo) custom symbolic functions can no longer take `ctx: torch.onnx.SymbolicContext` as the first argument.

#### Support for caffe2 has been removed (#129021)

- Remove Caffe2 handling from `onnx_unpack_quantized_weights` (#129021) 
- Remove `is_caffe2_aten_fallback` in `torch.onnx.symbolic_helper`

#### Some errors classes are removed

`CheckerError` and `InvalidExportOptionsError` are removed. Users can always catch `RuntimeError` to handle torch.onnx export errors.

---

## Deprecations

### Dynamo

- Remove `torch._dynamo.utils.CompileProfiler` (#135133)

### Export

- Deprecate `None` for specifying static dimensions in `dynamic_shapes` (#134877) 
 The use of `None` at the dimension-level for specifying dynamic shapes is now deprecated, and a user warning will be raised, so please use `Dim.STATIC` in its place. Specifying `None` for an entire input, or an entire program, is still supported. 

### Inductor

- aot_autograd: copy metadata from fw to bw nodes (#126573)
- deprecate `search_autotune_cache` (#133628)

### Releng

- Deprecate Python 3.8 support from CI/CD (#133621, #133624, #135245)

### ONNX

#### Supplying model keyword arguments to `torch.onnx.export` is deprecated (#131501) 
The ability to supply model keyword arguments as a final dictionary is deprecated. Users should use the `kwargs` parameter instead.

Deprecated: 
```python 
torch.onnx.export(model, (arg1, arg2, {“kwarg1”: …}))
```

Future: 
```python 
torch.onnx.export(model, (arg1, arg2), kwargs={“kwarg1”: …}) 
```

#### `torch.onnx.OperatorExportTypes` is deprecated (#131501)

The ability to supply `operator_export_type` in `torch.onnx.export()` is deprecated. Exported ONNX graphs will always use the ONNX opset domain. Options `ONNX_FALLTHROUGH`, `ONNX_ATEN` and `ONNX_ATEN_FALLBACK` are no longer supported. The `OperatorExportTypes` class will be removed in a future release.

#### The `training` option in `torch.onnx.export` is deprecated

Set the model training mode first before exporting instead.

Deprecated: 
```python 
torch.onnx.export(model, inputs, path, training=torch.onnx.TrainingMode.EVAL) 
```

Future: 
```python 
model = model.eval() 
torch.onnx.export(model, inputs, path) 
```

---

## New features

### Autograd frontend

- Add selective activation checkpoint support to `torch.utils.checkpoint` (#125795, #129262)

### Distributed

#### Flight Recorder with an analyzer 
 - Flight Recorder captures diagnostics information as collectives run- right now only for NCCL collectives. The captured diagnostic information is used to help root cause issues when jobs get stuck or timeout. An available analyzer script runs known heuristics using the collected data and attempts to automatically identify the underlying issue that caused the job to stall. (#110960, #113678, #114615, #114651, #114810, #114817, #115090, #115139, #115176, #115358, #115851, #118044, #118046, #118047, #119249, #119748, #119837, #120063, #120262, #120724, #120975, #122731, #126581, #126581, #126726, #128190, #128781, #128948, #129505, #130764, #131268, #133150, #133237, #133933, #133412, #134383, #134528, #134780, #134794)
 
#### c10d 
 - Enabled symmetricMemory-based, low contention intra-node `all-gather` and `reduce-scatter` (#130583)

### Dynamo

- Introduce `torch._dynamo.config.enable_compiler_collectives` for syncing compilation across ranks (#130935)

### Export

- `export_for_training [WIP/unstable]` (#129092, #130062, #134677, #135549) 
- Automatic dynamic shapes (#133620, #134486, #134702)

### Inductor

- [inductor] Add Triton template for Conv3D (#129518) 
- Autoheuristic: add config options for specifying for which optimizations to collect data, and for which optimizations to use learned heuristics (#130245) 
- Automatic horizontal fusion for Inductor ComboKernels (#131675) 
- Mode to emulate amp numerics (#131595) 
- [halide]The issue Add GPU support for the Halide backend adds the necessary functionality to enable GPU acceleration in PyTorch's Halide backend, improving compatibility with GPU-based computation and enhancing performance for specific workloads.(#127506) 
- [halide]The issue Enable bfloat16 support for the Halide backend introduces support for bfloat16 (bf16) data types in the Halide backend of PyTorch, expanding its capability to handle lower-precision computations and improving performance for models that benefit from mixed-precision training.(#129036) 
- [halide]The issue Support scan kernels in the Halide backend adds support for scan operations in PyTorch's Halide backend, enabling efficient reductions across multiple axes in tensors(#129035) 
- The issue Support adding a new inductor backend using PrivateUse1 enables the registration of a custom backend using the PrivateUse1 device type in PyTorch's inductor, facilitating backend extensions and new device types for specialized hardware.(#129953) 
- [halide]The issue Random number generation for the Halide backend introduces support for random number generation in PyTorch's Halide backend, enabling randomized operations for certain tensor computations that previously lacked support.(#130211) 
- [aoti] The issue Add packaging solution introduces a packaging solution for AOTInductor, allowing AOT-generated files to be packaged into a zipfile and loaded in Python. This feature supports the compilation and loading of precompiled models, enabling a more efficient workflow for distributed models (#129895) 
- Adds support for matrix decompositions when working with tensors that have unbacked sizes.(#128655) 
- Adds support for Intel GPUs by splitting reduction operations. (#129120) 
- Introduces a benchmark flag for inductor configuration, enhancing test workflows. (#129034) 
- Adds support for nested kernels in Triton when using indirect indexing. (#129223) 
- Introduces a composable kernel backend for ROCm-enabled devices in PyTorch's inductor. (#125453) 
- Adds support for mutating input tensors within CUDAGraph trees. (#129184) 
- Enables vectorization for bitwise operations in the C++ backend. (#129733) 
- Adds support for quantized linear GEMM templates with FP32 outputs. (#128825) 
- Extends GEMM template support to INT8 output with unary post-operation support. (#129048) 
- Adds support for binary fusion in GEMM templates for quantized linear operations. (#129103) 
- Adds support for AMX micro-GEMM kernel with int8 data type for quantized linear operations. (#129220) 
- Extends UserDefinedTritonKernel to support multiple outputs. (#129325) 
- Enables support for handling multiple outputs in FlexAttention operations. (#129344) 
- Introduces visualization methods for block masks in FlexAttention. (#129950) 
- Introduces the initial implementation of the B2B-GEMM pass with accompanying tests. (#129995) 
- Adds support for FX graph caching on AMD GPUs. (#130463) 
- Adds a GroupedSchedulerNode to handle nodes that need to be scheduled together in FSDP2. (#128568) 
- Adds support for flex decoding in FlexAttention's high-order programming (HOP). (#129415) 
- Adds partial masking support to FlexAttention. (#130415) 
- Adds a DeferredCudaGridLine wrapper for CUDA grid operations. (#129268) 
- Adds support for folding conv_bn with mixed data types in the post-grad phase. (#133968) 
- Enables OpenMP support in the inductor when using the Intel compiler on Linux. (#134973) 
- Enables the use of CUDA graphs even when there are unused CPU inputs. (#134749) 
- Adds support for MKLDNN convolution operations in the C++ wrapper for inductor. (#134475) 
- Introduces support for generalized linear operations using MKLDNN in the C++ wrapper. (#134783) 
- Extends support for quantized convolution operations in MKLDNN via the C++ wrapper. (#134795) 
- Adds support for unbacked symbolic integer (symint) divisors in variables and sizes. (#130595)

### nn

- Made `FlexAttention` API public (#130755) 
- Add `nn.Modules.set_submodule()` like `get_submodule` (#127714) 
- Add `nn.Buffer` like `nn.Parameter` (#125971)

### Optim

- Add an Adafactor impl (forloop and foreach) (#129905, #132336) 
- Add support for capturable optimizers on hpu and xpu (#132119)

### Optimizer Frontend

- Disable expandable segments checkpointing internally (#132048)

### Profiler

- [Profiler] Collect observer traces from C++ child threads (#128743) 
- [Profiler][XPU] Introduce kineto-based XPU profiler (#130811) 
- [Profiler] Add API for Dynamic Activity Toggling [2/n] (#133035) 
- [Memory Snapshot][Viz] Show event timestamps if collected (#132523) 
- [Memory Snapshot][Viz] Add Allocator Settings Tab (#132518) 
- [Profiler] Add kwargs to Kineto Traces (#130373)

### Python Frontend

- Add support for device extension autoloading (#127074) 
- Added support for sharing tensors on the meta device between processes (#129520) 
- add `__torch_function__` handler to `Tensor.get_device` cpp (#132567) 
- Add `torch.serialization.skip_data` context manager to create a metadata-only checkpoint (#134504) 
- Add `torch.serialization.safe_globals` context manager to work with weights_only serialization (#127939)

### Quantization

#### PT2E Numeric Debugger

- Preserve `_numeric_debug_handle` throguh deepcopy and re-export (#129287) 
- Add `numeric_debugger` top level APIs (#130643) 
- Update pt2e numeric debugger to use `node.meta["custom"]` field (#134040) 
- Fix output node's meta (#131706)

### Releng

- Split Build - Create a distribution of pytorch which is composed of a c++ component and mostly python component similar to jax and jaxlib (#129088, #127934, #126813, #129011, #129270, #129253, #129269, #129774, #132537, #124995, #134624) 
- Intel GPU enablement in CI/CD. Add prototype Linux Manywheel binary builds with better ATen operation coverage and improved torch.compile support (#129730, #129560, #128486, #130742, #130922, #133069, #132854, #129847, #134074, #134204, #134214, #134461, #134464, #134455, #134312, #133151, #124147) 
- Add prototype Linux Manywheel Python 3.13 binary builds (#130030, #132984, #133670)

### XPU

- Improve ATen operation coverage and support Intel Client GPUs in addition to Intel Data Center GPUs (#135833) 
- Enable Windows support and enable PyTorch wheel build for Windows (#135833, #133151, #134312) 
- Enable deterministic support for Intel GPU operations (#127277, #129864) 
- Support _GLIBCXX_USE_CXX11_ABI both 0 and 1 mode (#130110)

### Sparse Frontend

- Add pinned memory support to COO/CSR/CSC/BSR/BSC tensors (#129645) 
- Add MaskedTensor support to _is_any_true (#128574) 
- Add MaskedTensor support to *_like API, example: empty_like, etc. ops (#128637) 
- Add MaskedTensor passthrough: unfold, F.Unfold, F.Fold, stack (#125262)

### ONNX

#### The `dynamo=True` option and new export logic (#132530, #133743, #134304, #134782, #135378, #135399, #135786, #136162, #135134, #134976, #135367, #135418, #135591, #135520)

We introduce the `dynamo=True` option in `torch.onnx.export()`. This is recommended as a replacement for `torch.onnx.dynamo_export` starting in PyTorch 2.5.

Version 2.5: 
```python 
onnx_program = torch.onnx.export(model, inputs, kwargs=kwargs, dynamo=True) 
# Use the external_data option to save weights as external data 
onnx_program.save(“model.onnx”, external_data=True) 
# To save without initializers 
onnx_program.save(“model.onnx”, include_initializers=False, keep_initializers_as_inputs=True) 
```

`torch.onnx.export(model, args, dynamo=True, report=True, verify=True)` leverages `torch.export` and [ONNX IR](https://github.com/microsoft/onnxscript/blob/main/onnxscript/ir/README.md) to convert captured `ExportedProgram`s to ONNX efficiently and robustly. This new process reduces memory consumption by half compared to `dynamo_export` in 2.4, while preserving rich tensor shape and stack trace information in the ONNX graph. You can leverage the `report=True` option to obtain a conversion report in markdown format to diagnose any conversion issues. Set `verify=True` to verify the ONNX model numerically with ONNX Runtime.

When using `external_data=True` to save model weights as external data to the .onnx file, weights larger than 1 MB are now aligned at 64 KB addresses. This allows runtimes to memory-map weights for better memory efficiency during inference.

> [NOTE] 
> The `dynamo=True` option currently supports only ONNX opset 18. Future releases will expand support to newer opsets.

> [NOTE] 
> The `dynamo=True` option requires the latest versions of `onnxscript` and `onnx` packages.