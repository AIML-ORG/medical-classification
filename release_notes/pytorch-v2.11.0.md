# pytorch/pytorch v2.11.0

Source: https://github.com/pytorch/pytorch/releases/tag/v2.11.0

*(Features, highlights, backwards-incompatible changes, deprecations, and new features only.)*

# Highlights


 
 
 
 Added Support for Differentiable Collectives for Distributed Training
 
 
 
 
 FlexAttention now has a FlashAttention-4 backend on Hopper and Blackwell GPUs
 
 
 
 
 MPS (Apple Silicon) Comprehensive Operator Expansion
 
 
 
 
 Added RNN/LSTM GPU Export Support
 
 
 
 
 Added XPU Graph Support
 
 
 

For more details about these highlighted features, you can look at the [release blogpost](https://pytorch.org/blog/pytorch-2-11-release-blog/). Below are the full release notes for this release.

---

# Backwards Incompatible Changes


## Release Engineering

### Volta (SM 7.0) GPU support removed from CUDA 12.8 and 12.9 binary builds (#172598)

 Starting with PyTorch 2.11, the CUDA 12.8 and 12.9 pre-built binaries no longer include support for Volta GPUs (compute capability 7.0, e.g. V100). This change was necessary to enable updating to CuDNN 9.15.1, which is incompatible with Volta.

 Users with Volta GPUs who need CUDA 12.8+ should use the CUDA 12.6 builds, which continue to include Volta support. Alternatively, build PyTorch from source with Volta included in `TORCH_CUDA_ARCH_LIST`.

 Version 2.10:
 ```
 # CUDA 12.8 builds supported Volta (SM 7.0)
 pip install torch --index-url https://download.pytorch.org/whl/cu128
 # Works on V100
 ```

 Version 2.11:
 ```
 # CUDA 12.8 builds no longer support Volta
 # For V100 users, use CUDA 12.6 builds instead:
 pip install torch --index-url https://download.pytorch.org/whl/cu126
 ```

### PyPI wheels now ship with CUDA 13.0 instead of CUDA 12.x ([#172663](https://github.com/pytorch/pytorch/issues/172663), [announcement](https://dev-discuss.pytorch.org/t/transitioning-pypi-cuda-wheels-to-cuda-13-0-as-the-stable-release-2-11/3325))

 Starting with PyTorch 2.11, `pip install torch` on PyPI installs CUDA 13.0 wheels by default for both Linux x86_64 and Linux aarch64. Previously, PyPI wheels shipped with CUDA 12.x and only Linux x86_64 CUDA wheels were available on PyPI. Users whose systems have only CUDA 12.x drivers installed may encounter errors when running `pip install torch` without specifying an index URL.

 Additionally, CUDA 13.0 only supports Turing (SM 7.5) and newer GPU architectures on Linux x86_64. Maxwell and Pascal GPUs are no longer supported under CUDA 13.0. Users with these older GPUs should use the CUDA 12.6 builds instead.

 CUDA 12.6 and 12.8 binaries remain available via `download.pytorch.org`.

 Version 2.10:
 ```bash
 # PyPI wheel used CUDA 12.x
 pip install torch
 ```

 Version 2.11:
 ```bash
 # PyPI wheel now uses CUDA 13.0
 pip install torch

 # To get CUDA 12.8 wheels instead:
 pip install torch --index-url https://download.pytorch.org/whl/cu128

 # To get CUDA 12.6 wheels (includes Maxwell/Pascal/Volta support):
 pip install torch --index-url https://download.pytorch.org/whl/cu126
 ```


## Python Frontend

### `torch.hub.list()`, `torch.hub.load()`, and `torch.hub.help()` now default the `trust_repo` parameter to `"check"` instead of `None`. The `trust_repo=None` option has been removed. (#174101)

 Previously, passing `trust_repo=None` (or relying on the default) would silently download and run code from untrusted repositories with only a warning. Now, the default `"check"` behavior will prompt the user for explicit confirmation before running code from repositories not on the trusted list.

 Users who were explicitly passing `trust_repo=None` must update their code. Users who were already passing `trust_repo=True`, `trust_repo=False`, or `trust_repo="check"` are not affected.

 Version 2.10:
 ```python
 # Default trust_repo=None — downloads with a warning
 torch.hub.load("user/repo", "model")
 # Explicit None — same behavior
 torch.hub.load("user/repo", "model", trust_repo=None)
 ```

 Version 2.11:
 ```python
 # Default trust_repo="check" — prompts for confirmation if repo is not trusted
 torch.hub.load("user/repo", "model")
 # To skip the prompt, explicitly trust the repo
 torch.hub.load("user/repo", "model", trust_repo=True)
 ```


## torch.nn

### Add sliding window support to `varlen_attn` via `window_size`, making optional arguments keyword-only (#172238)

 The signature of `torch.nn.attention.varlen_attn` has changed: a `*` (keyword-only separator) has been inserted before the optional arguments. Previously, optional arguments like `is_causal`, `return_aux`, and `scale` could be passed positionally; they must now be passed as keyword arguments. A new `window_size` keyword argument has also been added.

 ```python
 # Before (2.10)
 output = varlen_attn(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, True, None, 1.0)

 # After (2.11) — pass as keyword argument
 output = varlen_attn(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, window_size=(-1, 0), return_aux=None, scale=1.0)
 ```

### Remove `is_causal` flag from `varlen_attn` (#172245)

 The `is_causal` parameter has been removed from `torch.nn.attention.varlen_attn`. Causal attention is now expressed through the `window_size` parameter: use `window_size=(-1, 0)` for causal masking, or `window_size=(W, 0)` for causal attention with a sliding window of size `W`. The default `window_size=(-1, -1)` corresponds to full (non-causal) attention.

 ```python
 # Before (2.10)
 output = varlen_attn(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, is_causal=True)

 # After (2.11) — use window_size instead
 output = varlen_attn(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, window_size=(-1, 0))
 ```


## Distributed

### `DebugInfoWriter` now honors `$XDG_CACHE_HOME` for its cache directory in C++ code, consistent with the Python side. Previously it always used `~/.cache/torch`. (#168232)

 This avoids issues where `$HOME` is not set or not writable. Users who relied on `~/.cache/torch` being used regardless of `$XDG_CACHE_HOME` may see debug info written to a different location.

 Version 2.10:
 ```
 # C++ DebugInfoWriter always wrote to ~/.cache/torch
 ```

 Version 2.11:
 ```
 # C++ DebugInfoWriter now respects $XDG_CACHE_HOME/torch (same as Python code)
 # Falls back to ~/.cache/torch if $XDG_CACHE_HOME is not set
 ```

### `DeviceMesh` now stores a process group registry (`_pg_registry`) directly, enabling `torch.compile` to trace through `get_group()`. (#172272)

 This may break code that skips `init_process_group`, loads a saved DTensor (constructing a DeviceMesh with no PGs), and later creates PGs separately — during `torch.compile` runtime the PG lookup will fail. Users should ensure process groups are initialized before constructing the DeviceMesh.

 Version 2.10:
 ```python
 # PGs resolved via global _resolve_process_group at runtime
 mesh = DeviceMesh(...) # PGs could be created later
 ```

 Version 2.11:
 ```python
 # PGs now stored on DeviceMesh._pg_registry; must exist at mesh creation
 dist.init_process_group(...) # Must be called before creating mesh
 mesh = DeviceMesh(...)
 ```


## Distributed (DTensor)

### `DTensor.to_local()` backward now converts `Partial` placements to `Replicate` by default when `grad_placements` is not provided. (#173454)

 Previously, calling `to_local()` on a `Partial` DTensor would preserve the `Partial` placement in the backward gradient, which could produce incorrect gradients when combined with `from_local()`. Now, the backward pass automatically maps `Partial` forward placements to `Replicate` gradient placements, matching the behavior of `from_local()`.

 Users who relied on the previous behavior (where `to_local()` backward preserved `Partial` gradients) may see different gradient values. To ensure correctness, explicitly pass `grad_placements` to `to_local()`.

 Version 2.10:
 ```python
 # Partial placement preserved in backward — could produce incorrect gradients
 local_tensor = partial_dtensor.to_local()
 ```

 Version 2.11:
 ```python
 # Partial → Replicate in backward by default (correct behavior)
 local_tensor = partial_dtensor.to_local()
 # Or explicitly specify grad_placements for full control:
 local_tensor = partial_dtensor.to_local(grad_placements=[Replicate()])
 ```

### `_PhiloxState.seed` and `_PhiloxState.offset` now return `torch.Tensor` instead of `int` (#173876)

 The DTensor RNG internal `_PhiloxState` class changed its `seed` and `offset` properties to return tensors instead of Python ints, and the setters now expect tensors. This makes the RNG state compatible with PT2 tracing (the previous `.item()` calls were not fake-tensor friendly).

 Code that directly reads `_PhiloxState.seed` or `_PhiloxState.offset` and treats them as ints will break. Call `.item()` to get the int value. When setting, wrap the value in a tensor.

 Version 2.10:
 ```python
 from torch.distributed.tensor._random import _PhiloxState

 philox = _PhiloxState(state)
 seed: int = philox.seed # returned int
 philox.offset = 42 # accepted int
 ```

 Version 2.11:
 ```python
 from torch.distributed.tensor._random import _PhiloxState

 philox = _PhiloxState(state)
 seed: int = philox.seed.item() # now returns Tensor; call .item() for int
 philox.offset = torch.tensor([42], dtype=torch.int64) # must pass Tensor
 ```


## ROCm

### caffe2 support is fully removed from ROCm PyTorch's hipify preprocessing. This is known as "hipify v2" behavior. (#174087, #174300, #174388, #174499, #175098)
#### hipify v1 background
When caffe2 and PyTorch were separate projects, the ROCm support strategies were different. For caffe2, all files and classes would be renamed following the pattern of CUDA to HIP, Cuda to Hip, cuda to hip, and so on. PyTorch did not rename classes, but would create new files following the same renaming pattern (e.g., aten/src/ATen/cuda/CUDABlas.h to aten/src/ATen/hip/HIPBlas.h). As a consequence, caffe2 had a distinct device backend named "HIP" (renamed from "CUDA") while ROCm PyTorch masquerades as the "cuda" device (`torch.empty(1, device="cuda")`). Once caffe2 and PyTorch projects were merged, this caused a mismatch between caffe2 expecting to use a "HIP" device while PyTorch expecting a "cuda" device. To alleviate this mismatch, "Masquerading" classes were created under aten/src/ATen/hip/impl.
- HIPAllocatorMasqueradingAsCUDA.h
- HIPCachingAllocatorMasqueradingAsCUDA.h
- HIPGuardImplMasqueradingAsCUDA.h
- HIPStreamMasqueradingAsCUDA.h
These classes were often transparently utilized during ROCm PyTorch's hipify preprocessing of source files. All files under c10/ and caffe2/ were hipified using the caffe2 renaming behavior, while all other "PyTorch" files used the other strategy. The Masquerading classes would replace their CUDA counterpart during hipify preprocessing. For example, c10/cuda/CUDAStream.h's CUDAStream would be replaced by aten/src/ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h's HIPStreamMasqueradingAsCUDA. These Masquerading classes call the underlying caffe2 code and create "HIP" devices, and the device would be reset to "cuda" by the Masquerading classes.
#### hipify v2 new behavior
Hipify v2 (#174087, #174300, #174388, #174499, #175098) makes the following changes:
- "Masquerading" classes are deprecated. Reworked to be thin shells around existing classes, for backward compatibility.
- Do not rename "CUDA" classes to "HIP". Only rename CUDA Runtime APIs. Files are still renamed out of place.
- Removes caffe2 work-arounds for HIP device versus CUDA device.
Great care has been taken to make this change backwards compatible. Though PyTorch today builds cleanly using hipify v2 behavior, downstream PyTorch extension projects that explicitly included Masquerading headers or called Masquerading APIs could be affected, resulting in failed builds. As an example, before backwards compatibility was realized, the xformers project had failed to build using the hipify v2 changes. A [PR demonstrates the changes that were initially necessary to work around the build failures,](https://github.com/facebookresearch/xformers/pull/1351) but such changes are no longer necessary after hipify v2 BC-breaking behavior was improved.


## torch.export

### `torch.export.export_for_training` has been removed (#171714)

 `export_for_training` was previously available as a separate API for exporting models while preserving training semantics. This function has been removed. Users should use `torch.export.export` instead, which returns the same graph as the previous `export_for_training`.


## ONNX

### **Remove the `fallback` option from `torch.onnx.export`** (#173189)

 The `fallback` parameter has been removed from `torch.onnx.export()`. Previously, when `fallback=True`, the exporter would automatically fall back to the legacy TorchScript-based exporter if the dynamo exporter failed. This fallback was removed because it was overly complicated, required different inputs, produced different models, and hid errors from the new exporter.

 **Migration:** Remove `fallback=True` (or `fallback=False`) from your `torch.onnx.export()` calls. If you need fallback behavior, implement it explicitly in your own code by catching exceptions and calling the legacy exporter separately.

 ```python
 # Before
 torch.onnx.export(model, args, "model.onnx", dynamo=True, fallback=True)

 # After
 torch.onnx.export(model, args, "model.onnx", dynamo=True)
 ```

### **Remove overload matching logic from the ONNX dispatcher** (#165083)

 The `custom_translation_table` parameter in `torch.onnx.export()` no longer accepts a list of functions for each torch op. Previously, users could pass a list of overloaded ONNX functions (e.g., one for float tensors, another for bool tensors), and the dispatcher would automatically select the correct overload based on input types. This complex type-matching logic has been removed because torchlib no longer uses overloads for the same opset version.

 The type of `custom_translation_table` changed from `dict[Callable, Callable | Sequence[Callable]]` to `dict[Callable, Callable]`. Passing a `Sequence` as a value now raises a `TypeError`.

 **Migration:** Provide a single function per operator instead of a list of overloads. If you need type-dependent behavior, handle it inside the single function.

 ```python
 # Before
 custom_translation_table = {
 torch.ops.aten.logical_and.default: [custom_impl_float, custom_impl_bool],
 }

 # After
 custom_translation_table = {
 torch.ops.aten.logical_and.default: custom_impl,
 }
 ```


## Quantization

### The PT2E quantization flow (`torch.ao.quantization.pt2e` and `torch.ao.quantization.quantizer`) has been removed from PyTorch and migrated to [torchao](https://github.com/pytorch/ao). (#169151)

 The following modules and classes have been removed:
 - `torch.ao.quantization.pt2e` (including `DuplicateDQPass`, `PortNodeMetaForQDQ`, export utils, graph utils, numeric debugger, lowering utilities)
 - `torch.ao.quantization.quantizer` (including `ComposableQuantizer`, `EmbeddingQuantizer`, `X86InductorQuantizer`, `XPUInductorQuantizer`, `XNNPACKQuantizer`, `QuantizationSpec`, `QuantizationAnnotation`, `QuantizationConfig`, etc.)

 Users relying on the PT2E quantization flow should migrate to the `torchao` package, which now hosts these APIs.

 Version 2.10:
 ```python
 from torch.ao.quantization.pt2e import prepare_pt2e, convert_pt2e
 from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
 ```

 Version 2.11:
 ```python
 # Install torchao: pip install torchao
 from torchao.quantization.pt2e import prepare_pt2e, convert_pt2e
 from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import X86InductorQuantizer
 ```

---

# Deprecations


## Linear Algebra

- The MAGMA backend for linear algebra operations is now deprecated and will be removed in a future release. Setting `torch.backends.cuda.preferred_linalg_library("magma")` or retrieving a previously-set MAGMA preference will now issue a deprecation warning. cuSOLVER remains the default backend. (#172823)

 If you see any errors when using cuSOLVER that did not occur with MAGMA, please file an issue on GitHub. To silence the warning, stop explicitly selecting the MAGMA backend:

 Version 2.10:
 ```python
 # No warning
 torch.backends.cuda.preferred_linalg_library("magma")
 ```

 Version 2.11:
 ```python
 # Issues a deprecation warning — remove this call to use the default cuSOLVER backend
 torch.backends.cuda.preferred_linalg_library("magma")
 ```

- `torch.linalg.svd` no longer dispatches to MAGMA. The MAGMA backend is deprecated and cuSOLVER is now used unconditionally, providing significant speedups (2x–400x depending on matrix size and batch dimensions). (#172824)

 Previously, setting `torch.backends.cuda.preferred_linalg_library("magma")` would route SVD through MAGMA. This setting is now ignored for SVD, and cuSOLVER is always used.

 Version 2.10:
 ```python
 torch.backends.cuda.preferred_linalg_library("magma")
 U, S, Vh = torch.linalg.svd(x) # Uses MAGMA
 ```

 Version 2.11:
 ```python
 # MAGMA preference is ignored; cuSOLVER is always used
 U, S, Vh = torch.linalg.svd(x) # Uses cuSOLVER
 ```

- `torch.linalg.solve_triangular` and `torch.triangular_solve` no longer dispatch to MAGMA on CUDA. cuBLAS is now used unconditionally, providing speedups of 2x–24x for most matrix sizes (small matrices may see minor regressions of ~0.6x). (#174109)

 Version 2.10:
 ```python
 torch.backends.cuda.preferred_linalg_library("magma")
 torch.linalg.solve_triangular(A, B, upper=False) # Uses MAGMA
 ```

 Version 2.11:
 ```python
 # MAGMA preference is ignored; cuBLAS is always used
 torch.linalg.solve_triangular(A, B, upper=False) # Uses cuBLAS
 ```

- `torch.linalg.lstsq` no longer dispatches to MAGMA. cuSOLVER/cuBLAS are now used unconditionally, providing speedups of 1.7x–620x depending on matrix size and batch dimensions. (#174779)

 Version 2.10:
 ```python
 torch.backends.cuda.preferred_linalg_library("magma")
 result = torch.linalg.lstsq(A, B) # Uses MAGMA
 ```

 Version 2.11:
 ```python
 # MAGMA preference is ignored; cuSOLVER/cuBLAS is always used
 result = torch.linalg.lstsq(A, B) # Uses cuSOLVER/cuBLAS
 ```


## Distributed

### `torch.distributed.symmetric_memory.enable_symm_mem_for_group` is deprecated. The store can be retrieved directly via `ProcessGroup.getStore()` in C++, making this call unnecessary. (#172163)

 Version 2.10:
 ```python
 from torch.distributed.symmetric_memory import enable_symm_mem_for_group
 enable_symm_mem_for_group(group)
 ```

 Version 2.11:
 ```python
 # No longer needed — store is accessed directly from the ProcessGroup
 ```

---

# New features


## Python Frontend
- Added `native_handle` property to `torch.Stream`, providing a unified way to retrieve the backend-specific opaque stream handle (e.g., `cudaStream_t` for CUDA, `sycl::queue*` for XPU). This is useful for passing stream handles to third-party libraries such as Triton. (#171040)

 ```python
 stream = torch.accelerator.current_stream()
 handle = stream.native_handle # backend-specific stream handle
 ```

## Autograd
- Add `Function.clear_saved_tensors_on_access` class attribute to automatically free saved tensors after they are accessed (#173833)

## torch.nn
- Add mechanism to restore default flash attn impl after `activate_flash_attention_impl` (#169866)
- Add `scale` for softmax to varlen attn (#171199)

## Distributed
- Add `start_method` option to `torch.distributed.debug.start_debug_server` to select the multiprocessing start method (`fork`, `spawn`, or `forkserver`), enabling CUDA-safe server startup (#173196)
- Add support for periodic dumping in `torch.distributed.debug` (#174808)
- Non-functional collectives (e.g. `torch.distributed.all_gather`) now automatically work with `FakeTensorMode` — meta implementations are registered at `import torch` time (#162119)
- Implement NCCL 2.29 one-sided APIs for symmetric memory (#172425)
- Bind `SymmetricMemory` as a torch class for use in op definitions (#174019)
- Enable `torchcomms` `_BackendWrapper` shim layer in c10d (#174202)
- Expose SymmetricMemory window API (#170740)

## CUDA
- Make (pinned) host memory allocations work with memory pools. (#167507)
- Make large segment size configurable for allocation performance tuning (esp. re: Expandable Segments). (#172056)

## MPS
- Async error reporting from GPU operations (#170002, #170050)
 ```python
 import torch
 x = torch.rand(10, 1, 10, device='mps')
 y = x[:, [1]]
 torch.mps.synchronize() # will raise index out of bounds error
 ```
- Added support for Metal 4 (#172229, #172230)

## ROCm
- Expose device properties `clock_rate`, `memory_clock_rate`, `memory_bus_width`, `memory_per_block`, `shared_memory_per_block`. (#170572)
- Support for device-side assertions via `TORCH_USE_HIP_DSA`. (#172679)
- Attention operator support on gfx1151/1152/1153 via AOTriton 0.11.2b update (#174105)
- Enable scaled group mm on gfx950. (#173737)
- Enable group gemm on gfx90a. (#169356)
- Enable MIOpen backend for CTC Loss. (#170749)
- Add hipsparseSpSV and hipsparseSpSM support for triangular solve. (#171097)
- Support for PyTorch's StaticCudaLauncher, which provides static compilation and launching of Triton kernels. (#166492)

## XPU
- Introduce XPUGraph, a runtime optimization feature designed to reduce kernel host overhead on XPU devices, detail in: [design](https://github.com/pytorch/pytorch/issues/162143) and [usage](https://docs.pytorch.org/docs/2.11/xpu.html#graphs). (#166285, #174041, #174351, #174059, #174046, #166843)

## torch.compile
#### Dynamo
- `torch.compile` now supports tracing through `contextlib.ExitStack` and `contextlib.suppress` context managers, allowing code that uses these patterns to be compiled without graph breaks (#146506, #147990)
- Added `torch._dynamo.config.ignore_logging_functions` config to skip arbitrary logging callables during tracing without causing graph breaks. Add functions to this set to have Dynamo treat them as no-ops during compilation (#168913)
- Added `TORCH_DYNAMO_AUTOMATIC_DYNAMIC_SHAPES=0` environment variable to globally disable automatic dynamic shapes without modifying Python code (#172334)
- Added `TORCH_COMPILE_OVERRIDE_BACKENDS` environment variable for per-graph backend override, enabling binary search to find problematic compiled graphs. Supports filter syntax like `">10:eager"` or `"0-5:aot_eager;6-10:inductor"` (#172411)
- Added initial support for `torch._dynamo.decorators.leaf_function`, which allows annotating functions as leaf operations that Dynamo and AOTAutograd will not trace into (#170471)
- Added support for tracing backward hooks on intermediate tensors, fixing cases where `register_hook` on non-leaf tensors would fail under `torch.compile` (#172126)
#### Inductor
- FlexAttention supports deterministic mode, wired through both Flex and Flash backends (#173126)
- Added range-based autotuning for custom ops, enabling selection of optimal implementations based on runtime tensor dimension values with per-range benchmarking and automatic `torch.cond` dispatch generation (#167617)
- FlexAttention: Added support for low precision K/V inputs in compiled mode. Keys and Values can now be in lower precision than Queries for memory efficiency (#171761)
- Added native `ldexp` lowering with `libdevice.ldexp` (CUDA) and `std::ldexp` (CPU) codegen (#171721)
- Inductor now supports `pin_memory` for `torch.empty` (#172578)
- Exposed `triton_meta` to TritonTemplate `maybe_append_choice` API for custom template development (#174292)
- Added Async Pipelined Autotuning for `max-autotune-gemm`, which overlaps autotuning with lowering/scheduling in a subprocess to reduce compilation overhead (#170407)
- FlexFlash: Added BlockSparse backward pass, dynamic shapes, and backward score-mod support (#170397, #170611, #171465)
- Added FP8 `(BlockWise128x128, BlockWise1x128)` scaling support in Inductor Triton templates (#170748)
- Autochunker: Added gradient accumulation support and ability to override number of chunks (#171359, #171477)
- Added NVGEMM backend for GEMM operations using NVIDIA's native matmul library, with support for BMM, grouped GEMM, scaled MM, dynamic shapes (#171205, #171362, #172280, #172283, #172378, #172388, #172391, #172402, #172417, #172525, #172582, #172607, #174827)

## torch.export
- Add nested tensor serialization support for `torch.export` (#174720)
- RNN modules (LSTM, GRU, etc.) can now be exported on GPUs (#169710)
- Add patch to enable tracing LSTM with dynamic shapes (#168095)

## ONNX
- Added `ExportableModule` wrapper for ONNX export (#170810)
- Added `InputObserver` to infer dynamic shapes for export (#172838)
- Add a parameter to force the first dimension to be dynamic in InputObserver.infer_dynamic_shapes (#173533)
- Implement while_loop (#162645)
- Add invoke_subgraph HOP export support (#174283)
- Expose ONNXProgram.rename_axes for renaming dims (#172032)
- Support custom empty tensor shapes in `InputObserver` for multimodal LLM export (#174964)

## Foreach
- Added `torch.linalg._powsum` and `torch._foreach_powsum` as fused kernels that compute `sum(abs(x)**ord)` (equivalent to `vector_norm` without the root extraction) (#172685)