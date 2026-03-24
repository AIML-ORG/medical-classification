# pytorch/pytorch v2.8.0

Source: https://github.com/pytorch/pytorch/releases/tag/v2.8.0

*(Features, highlights, backwards-incompatible changes, deprecations, and new features only.)*

# Highlights
 
 
 Unstable 
 
 
 torch::stable::Tensor 
 
 
 High-performance quantized LLM inference on Intel CPUs with native PyTorch 
 
 
 Experimental Wheel Variant Support 
 
 
 Inductor CUTLASS backend support 
 
 
 Inductor Graph Partition for CUDAGraph 
 
 
 Control Flow Operator Library 
 
 
 HuggingFace SafeTensors support in PyTorch Distributed Checkpointing 
 
 
 SYCL support in PyTorch CPP Extension API 
 
 
 A16W4 on XPU Device 
 
 
 Hierarchical compilation with torch.compile 
 
 
 Intel GPU distributed backend (XCCL) support 
 
 

For more details about these highlighted features, you can look at the [release blogpost](https://pytorch.org/blog/pytorch-2-8/).
Below are the full release notes for this release.

---

# Backwards Incompatible Changes


## CUDA Support
### Removed support for Maxwell and Pascal architectures with CUDA 12.8 and 12.9 builds (#157517, #158478, #158744)
Due to binary size limitations, support for sm50 - sm60 architectures with CUDA 12.8 and 12.9 has
been dropped for the 2.8.0 release. If you need support for these architectures, please utilize
CUDA 12.6 instead.


## Python Frontend
### Calling an op with an input dtype that is unsupported now raises `NotImplementedError` instead of `RuntimeError` (#155470)
Please update exception handling logic to reflect this.

In 2.7.0
```
try:
 torch.nn.Hardshrink()(torch.randint(0, 5, (10,)))
except RuntimeError:
 ...
```

In 2.8.0
```
try:
 torch.nn.Hardshrink()(torch.randint(0, 5, (10,)))
except NotImplementedError:
 ...
```

### Added missing in-place on view check to custom `autograd.Function` (#153094)

In 2.8.0, if a custom `autograd.Function` mutates a view of a leaf requiring grad,
it now properly raises an error. Previously, it would silently leak memory.
```
 class Func(torch.autograd.Function):
 @staticmethod
 def forward(ctx, inp):
 inp.add_(1)
 ctx.mark_dirty(inp)
 return inp

 @staticmethod
 def backward(ctx, gO):
 pass

 a = torch.tensor([1.0, 2.0], requires_grad=True)
 b = a.view_as(a)
 Func.apply(b)
```
Output:

Version 2.7.0
```
Runs without error, but leaks memory
```
Version 2.8.0
```
RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation
```

### An error is now properly thrown for the out variant of `tensordot` when called with a `requires_grad=True` tensor (#150270)

Please avoid passing an out tensor with `requires_grad=True` as gradients cannot be
computed for this tensor.

In 2.7.0
```
a = torch.empty((4, 2), requires_grad=True)
b = torch.empty((2, 4), requires_grad=True)
c = torch.empty((2, 2), requires_grad=True)
# does not error, but gradients for c cannot be computed
torch.tensordot(a, b, dims=([1], [0]), out=c)
```

In 2.8.0
```
a = torch.empty((4, 2), requires_grad=True)
b = torch.empty((2, 4), requires_grad=True)
c = torch.empty((2, 2), requires_grad=True)
torch.tensordot(a, b, dims=([1], [0]), out=c)
# RuntimeError: tensordot(): the 'out' tensor was specified and requires gradients, and
# its shape does not match the expected result. Either remove the 'out' argument, ensure
# it does not require gradients, or make sure its shape matches the expected output.
```


## torch.compile
### Specialization of a tensor shape with `mark_dynamic` applied now correctly errors (#152661)

Prior to 2.8, it was possible for a guard on a symbolic shape to be incorrectly
omitted if the symbolic shape evaluation was previously tested with guards
suppressed (this often happens within the compiler itself). This has been fixed
in 2.8 and usually will just silently "do the right thing" and add the correct
guard. However, if the new guard causes a tensor marked with `mark_dynamic` to become
specialized, this can result in an error. One workaround is to use
`maybe_mark_dynamic` instead of `mark_dynamic`.

See the discussion in issue #157921 for more
context.

Version 2.7.0
```python
import torch

embed = torch.randn(2, 8192)
x = torch.zeros(8192)

torch._dynamo.mark_dynamic(x, 0)

@torch.compile
def f(embedding_indices, x):
 added_tokens_mask = torch.where(x > 10000, 1, 0)
 ei = torch.narrow(embedding_indices, 1, 0, x.size(0))
 return ei.clone()

f(embed, x)
```

Version 2.8.0
```python
import torch

embed = torch.randn(2, 8192)
x = torch.zeros(8192)

torch._dynamo.maybe_mark_dynamic(x, 0)

@torch.compile
def f(embedding_indices, x):
 added_tokens_mask = torch.where(x > 10000, 1, 0)
 ei = torch.narrow(embedding_indices, 1, 0, x.size(0))
 return ei.clone()

f(embed, x)
```

### Several config variables related to `torch.compile` have been renamed or removed
- Dynamo config variable `enable_cpp_framelocals_guard_eval` has changed to no longer have any effect (#151008).
- Inductor config variable `rocm.n_max_profiling_configs` is deprecated (#152341).
Instead, use ck-tile based configs `rocm.ck_max_profiling_configs` and
`rocm.ck_tile_max_profiling_configs`.
- Inductor config variable `autotune_fallback_to_aten` is deprecated (#154331).
Inductor will no longer silently fall back to `ATen`. Please add `"ATEN"` to
`max_autotune_gemm_backends` for the old behavior.
- Inductor config variables `use_mixed_mm` and `mixed_mm_choice` are deprecated (#152071). Inductor now supports prologue fusion, so there is no need for
special cases now.
- Inductor config setting `descriptive_names = False` is deprecated (#151481). Please use one of the other available
options: `"torch"`, `"original_aten"`, or `"inductor_node"`.
- `custom_op_default_layout_constraint` has moved from inductor config to functorch config (#148104). Please reference it via
`torch._functorch.config.custom_op_default_layout_constraint` instead of
`torch._inductor.config.custom_op_default_layout_constraint`.
- AOTI config variable `emit_current_arch_binary` is deprecated (#155768).
- AOTI config variable `aot_inductor.embed_cubin` has been renamed to `aot_inductor.embed_kernel_binary` (#154412).
- AOTI config variable `aot_inductor.compile_wrapper_with_O0` has been renamed to `compile_wrapper_opt_level` (#148714).

### Added a stricter aliasing/mutation check for `HigherOrderOperator`s (e.g. `cond`), which will explicitly error out if alias/mutation among inputs and outputs is unsupported (#148953, #146658).

For affected `HigherOrderOperator`s, add `.clone()` to aliased outputs to address this.

Version 2.7.0
```python
import torch

@torch.compile(backend="eager")
def fn(x):
 return torch.cond(x.sum() > 0, lambda x: x, lambda x: x + 1, [x])

fn(torch.ones(3))
```

Version 2.8.0
```python
import torch

@torch.compile(backend="eager")
def fn(x):
 return torch.cond(x.sum() > 0, lambda x: x.clone(), lambda x: x + 1, [x])

fn(torch.ones(3))
```

### `guard_or_x` and `definitely_x` have been consolidated (#152463)
We removed `definitely_true` / `definitely_false` and associated APIs, replacing them with
`guard_or_true` / `guard_or_false`, which offer similar functionality and can be used to
achieve the same effect. Please migrate to the latter.

Version 2.7.0
```python
from torch.fx.experimental.symbolic_shapes import definitely_false, definitely_true

...
if definitely_true(x):
 ...

if definitely_false(y):
 ...
```

Version 2.8.0
```python
from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true

...
if guard_or_false(x):
 ...

# alternatively: if guard_or_false(torch.sym_not(y))
if not guard_or_true(y):
 ...
```


## torch.export
### `torch.export.export_for_inference` has been removed in favor of `torch.export.export_for_training().run_decompositions()` (#149078)

Version 2.7.0
```python
import torch

...
exported_program = torch.export.export_for_inference(mod, args, kwargs)
```

Version 2.8.0
```python
import torch

...
exported_program = torch.export.export_for_training(
 mod, args, kwargs
).run_decompositions(decomp_table=decomp_table)
```

### Switched default to `strict=False` in `torch.export.export` and `export_for_training` (#148790, #150941)

This differs from the previous release default of `strict=True`. To revert to the old default
behavior, please explicitly pass `strict=True`.

Version 2.7.0
```python
import torch

# default behavior is strict=True
torch.export.export(...)
torch.export.export_for_training(...)
```

Version 2.8.0
```python
import torch

# strict=True must be explicitly passed to get the old behavior
torch.export.export(..., strict=True)
torch.export.export_for_training(..., strict=True)
```


## ONNX
### Default opset in `torch.onnx.export` is now 18 (#156023)

When `dynamo=False`, the default ONNX opset version has been updated from 17 to 18. Users can set `opset_version` to explicitly select an opset version.

Version 2.7

```py
# opset_version=17
torch.onnx.export(...)
```

Version 2.8

```py
# To preserve the original behavior
torch.onnx.export(..., opset_version=17)

# New: opset_version=18
torch.onnx.export(...)
```

### The `JitTraceConvertStrategy` has been removed (#152556)

Support for JIT traced and scripted modules in the ONNX exporter when `dynamo=True` has been removed. You are encouraged to export an nn.Module directly, or create an `ExportedProgram` using `torch.export` before exporting to ONNX.

### `onnxscript>=0.3.1` is required for the `dynamo=True` option (#157017)

You must upgrade `onnxscript` to version 0.3.1 or higher for it to be compatible with PyTorch 2.8.


## Build Frontend
### Removed the `torch/types.h` include from `Dispatcher.h` (#149557)
This can cause build errors in C++ code that implicitly relies on this include (e.g. very old versions of `torchvision`).

Note that `Dispatcher.h` does not belong as an include from `torch/types.h` and was only present as a
short-term hack to appease `torchvision`. If you run into `torchvision` build errors, please
update to a more recent version of `torchvision` to resolve this.

### Upgraded `DLPack` to 1.0 (#145000)
As part of the upgrade, some of the `DLDeviceType` enum values have been renamed. Please switch
to the new names.

Version 2.7.0
```
from torch.utils.dlpack import DLDeviceType

d1 = DLDeviceType.kDLGPU
d2 = DLDeviceType.kDLCPUPinned
...
```

Version 2.8.0
```
from torch.utils.dlpack import DLDeviceType

d1 = DLDeviceType.kDLCUDA # formerly kDLGPU
d2 = DLDeviceType.kDLCUDAHost # formerly kDLCPUPinned
...
```

### NVTX3 code has been moved from `cmake/public/cuda.cmake` to `cmake/Dependencies.cmake` (#151583)

This is a BC-breaking change for the build system interface. Downstream projects that previously got NVTX3 through `cmake/public/cuda.cmake`
(i.e.. calling `find_package(TORCH REQUIRED)`) will now need to explicitly configure NVTX3 support in the library itself (i.e. use `USE_SYSTEM_NVTX=1`).
The change is to fix the broken behavior where downstream projects couldn't find NVTX3 anyway due to the `PROJECT_SOURCE_DIR` mismatch.

Version 2.7.0:
- A downstream project using `-DUSE_SYSTEM_NVTX` would be able to find NVTX3 and `torch::nvtx3` via PyTorch's `cmake/public/cuda.cmake` logic.
- A downstream project NOT using `-DUSE_SYSTEM_NVTX` would encounter build errors with CUDA 12.8 or above.

Version 2.8.0:
- A downstream project using `-DUSE_SYSTEM_NVTX` will not be able to find NVTX3 or `torch::nvtx3` via PyTorch's `cmake/public/cuda.cmake`. The downstream project now needs to explicitly find NVTX3 and torch::nvtx3 by implementing the same logic in PyTorch's `cmake/Dependences.cmake`.
- A downstream project NOT using `-DUSE_SYSTEM_NVTX` will proceed building without NVTX unless another part of the build process re-enables NVTX.

---

# Deprecations
### MPS support for MacOS Ventura will be removed in 2.9
PyTorch 2.8 is the last release that will support GPU acceleration on MacOS Ventura. In the next
release (2.9), MacOS Sonoma (released in Sept. 2023) or above will be required to use the MPS
backend.

### `torch.ao.quantization` is deprecated and will be removed in 2.10 (#153892)
To migrate:
- Eager mode quantization (`torch.ao.quantization.quantize`, `torch.ao.quantization.quantize_dynamic`)
 - Weight-only and dynamic quantization: use `torchao` eager mode `quantize_`.
 - Static quantization: use `torchao` PT2E quantization.
- FX graph mode quantization (`torch.ao.quantization.quantize_fx.prepare_fx`, `torch.ao.quantization.quantize_fx.convert_fx`): use `torchao` PT2E quantization (`torchao.quantization.quantize_pt2e.prepare_pt2e`, `torchao.quantization.quantize_pt2e.convert_pt2e`).

Note that PT2E quantization has been migrated to `torchao` (https://github.com/pytorch/ao/tree/main/torchao/quantization/pt2e). See https://github.com/pytorch/ao/issues/2259 and https://docs.pytorch.org/ao/main/quick_start.html#pytorch-2-export-quantization for more details.

### The `dynamo=False` (current default) option for `torch.onnx.export` is deprecated (#152478, #155580)

The default will be `dynamo=True` starting from PyTorch 2.9. You are encouraged to migrate to use the `dynamo=True` option in `torch.onnx.export`. This flag makes `torch.export.export` the default export path, replacing `TorchScript`.

To maintain the old behavior, set `dynamo=False` explicitly. You are encouraged to also experiment with the `fallback=True` option that will make the exporter fall back to the `dynamo=False` path if there are errors.

---

# New Features

## CUDA
- Support capture of event record and wait in CUDAGraphs for timing (#155372)


## torch.compile
#### Dynamo
- Added support for hierarchical compilation via `nested_compile_region` (#156449)
- Allow guards to be dropped with custom filter functions via `guard_filter_fn` (#150936)
- Added `dont_skip_tracing` decorator to skip over most Dynamo `skipfiles` rules (#150586)

#### Inductor
- Added support for mapping a Dynamo graph to multiple different Inductor graphs, which can be optimized separately (#147648, #147038)


## torch.export
- Introduced [`draft-export`](https://docs.pytorch.org/docs/main/export/draft_export.html), an export variant designed to consistently produce a graph and generate a debugging report of issues encountered during tracing (#152637, #153219, #149465, #153627, #154190, #155744, #150876, #150948, #151051, #151065, #150809, #151797)


## Ahead-Of-Time Inductor (AOTI)
- Added support for `TorchBind` objects (#150196, #154265)
- Added config variable `aot_inductor.model_name_for_generated_files` for specifying model name (#154129)


## MPS
- `MPSInductor`: `torch.compile` for Apple GPUs (#150121, #149342, #151449, #151754, #149687, #149180, #149221, #153598, #152788, #153787, #152214, #151152, #155891, #154578, #151272, #151288, #153997, #151871, #153362, #156566, #150661, #153582)


## ONNX
- Added new strategy `draft_export` (#147529, [docs](https://docs.pytorch.org/docs/main/draft_export.html)) to provide debugging information upon data-dependent / constraint errors when obtaining an `ExportedProgram` with `torch.onnx.export`

- Added support for symbolic operators in the `dynamo=True` export path (#148905, #149678, #150038, [docs](https://docs.pytorch.org/docs/main/onnx_ops.html#symbolic-operators)). Two operators `torch.onnx.ops.symbolic` and `torch.onnx.ops.symbolic_multi_out` are defined to allow you to create symbolic ONNX operators directly in your PyTorch models. You can use them in a `forward` method:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
 # Optionally use is_in_onnx_export to control the behavior during onnx export

 if torch.onnx.is_in_onnx_export():
 # Create a symbolic ONNX operator with the name "CustomOp" in the "custom_domain" domain.
 # The output tensor will have the specified dtype and shape
 return torch.onnx.ops.symbolic(
 "custom_domain::CustomOp",
 (x,),
 dict(attr_key="attr_value"),
 dtype=x.dtype,
 shape=x.shape,
 version=1,
 )
 else:
 return x
```


## Python Frontend
- Added Generalized Pareto Distribution (GPD) (#135968)


## Quantization
- Introduced `torch.float4_e2m1fn_x2` dtype (#148791)


## XPU
- Support Intel distributed backend (XCCL) (#141856)
- Support SYCL kernels through C++ extension (#132945)