# pytorch/pytorch v2.7.0

Source: https://github.com/pytorch/pytorch/releases/tag/v2.7.0

*(Features, highlights, backwards-incompatible changes, deprecations, and new features only.)*

# Highlights
 
 
 Beta 
 
 Prototype 
 
 
 
 Torch.Compile support for Torch Function Modes
 
 NVIDIA Blackwell Architecture Support
 
 
 
 Mega Cache
 
 PyTorch Native Context Parallel
 
 
 
 
 
 Enhancing Intel GPU Acceleration
 
 
 
 
 
 FlexAttention LLM first token processing on X86 CPUs 
 
 
 
 
 
 FlexAttention LLM throughput mode optimization on X86 CPUs
 
 
 
 
 
 Foreach Map
 
 
 
 
 
 Flex Attention for Inference
 
 
 
 
 
 Prologue Fusion Support in Inductor
 
 
 

For more details about these highlighted features, you can look at the [release blogpost](https://pytorch.org/blog/pytorch2-7/).
Below are the full release notes for this release.

---

# Backwards Incompatible Changes

### Dropped support for Triton < 2.2.0. Removed Support for CUDA 12.4, Anaconda in CI/CD.
- Removed CUDA 12.4 support in CI/CD in favor of 12.8 (#148895, #142856, #144118, #145566, #145844, #148602, #143076, #148717)
- Removed Anaconda support in CI/CD (#144870, #145015, #147792)
- Dropped support for Triton < 2.2.0 (versions without ASTSource) (#143817)

### C++ Extensions `py_limited_api=True` is now built with `-DPy_LIMITED_API` (#145764)

We formally began respecting the `py_limited_api=True` kwarg in 2.6 and stopped linking `libtorch_python.so` when the flag was specified, as libtorch_python.so does not guarantee using APIs from from the stable Python limited API. In 2.7, we go further by specifying the `-DPy_LIMITED_API` flag which will enforce that the extension is buildable with the limited API. As a result of this enforcement, **custom extensions that set `py_limited_api=True` but do not abide by the limited API may fail to build**. For an example, see #152243.

This is strictly better behavior as it is sketchy to claim CPython agnosticism without enforcing with the flag. If you run into this issue, please ensure that the extension you are building does not use any APIs which are outside of the Python limited API, e.g., `pybind`.

### Change `torch.Tensor.new_tensor()` to be on the given Tensor's device by default (#144958)

This function was always creating the new Tensor on the "cpu" device and will now use the same device as the current Tensor object. This behavior is now consistent with other `.new_*` methods.

### Use Manylinux 2.28 and CXX11_ABI=1 for future released Linux wheel builds. 
With Migration to manylinux_2_28 (AlmaLinux 8 based), we can no longer support OS distros with glibc2_26. These include popular Amazon Linux 2 and CentOS 7. (#143423, #146200, #148028, #148135, #148195, #148129)

### `torch.onnx.dynamo_export` now uses the ExportedProgram logic path (#137296)

Users using the `torch.onnx.dynamo_export` API may see some `ExportOptions` become
unsupported due to an internal switch to use `torch.onnx.export(..., dynamo=True)`: `diagnostic_options`, `fake_context` and `onnx_registry` are removed/ignored by `ExportOptions`. Only `dynamic_shapes` is retained.

Users should move to use the `dynamo=True` option on `torch.onnx.export` as
`torch.onnx.dynamo_export` is now deprecated. Leverage the [`dynamic_shapes`](https://pytorch.org/docs/stable/export.html#torch.export.export) argument in `torch.onnx.export` for specifying dynamic shapes on the model.

Version 2.6.0

```python
torch.onnx.dynamo_export(model, *args, **kwargs)
```

Version 2.7.0

```python
torch.onnx.export(model, args, kwargs=kwargs, dynamo=True)
```

### Finish deprecation of `LRScheduler.print_lr()` along with the `verbose` kwarg to the LRScheduler constructor. (#147301)

Both APIs have been deprecated since 2.2. Please use `LRScheduler.get_last_lr()` to access the learning rate instead.`print_lr` and `verbose` were confusing, not properly documented and were little used, as described in #99270, so we deprecated them in 2.2. Now, we complete the deprecation by removing them completely. To access and print the learning rate of a LRScheduler:

Version 2.6.0
```python
optim = ...
lrsched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, verbose=True)
// lrsched will internally call print_lr() and print the learning rate 
```

Version 2.7.0
```python
optim = ...
lrsched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)
print(lrsched.get_last_lr())
```

### libtorch_python.so symbols are now invisible by default on all platforms except Apple (#142214)
Previously, the symbols in libtorch_python.so were exposed with default visibility. We have transitioned to being more intentional about what we expose as public symbols for our python API in C++. After #142214, public symbols will be marked explicitly while everything else will be hidden. Some extensions using private symbols will see linker failures with this change.

### Please use `torch.export.export` instead of `capture_pre_autograd_graph` to export the model for pytorch 2 export quantization (#139505)

`capture_pre_autograd_graph` was a temporary API in `torch.export`. Since now we have a better longer term API: `export` available, we can deprecate it.

Version 2.6.0
```python
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
 XNNPACKQuantizer,
 get_symmetric_quantization_config,
)
quantizer = XNNPACKQuantizer().set_global(
 get_symmetric_quantization_config()
)
m = capture_pre_autograd_graph(m, *example_inputs)
m = prepare_pt2e(m, quantizer)
```

Version 2.7.0
```python
from torch.export import export
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
# please get xnnpack quantizer from executorch (https://github.com/pytorch/executorch/)
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
 XNNPACKQuantizer,
 get_symmetric_quantization_config,
)
quantizer = XNNPACKQuantizer().set_global(
 get_symmetric_quantization_config()
)
m = export(m, *example_inputs)
m = prepare_pt2e(m, quantizer)
```

### New interface for `torch.fx.passes.graph_transform_observer.GraphTransformObserver` to enable Node Level provenance tracking (#144277)
We now track a mapping between the nodes in the pre-grad and post-grad graph. See the issue for an example frontend to visualize the transformations. To update your `GraphTransformObserver` subclasses, instead of overriding `on_node_creation` and `on_node_erase`, there are new functions `get_node_creation_hook`, `get_node_erase_hook`, `get_node_replace_hook` and `get_deepcopy_hook`. These are registered on the `GraphModule` member of the `GraphTransformObserver` upon entry and exit of a `with` block

Version 2.6.0

```python
class MyPrintObserver(GraphTransformObserver):
 def on_node_creation(self, node: torch.fx.Node):
 print(node)
```
Version 2.7.0
```python
class MyPrintObserver(GraphTransformObserver):
 def get_node_creation_hook(self):
 def hook(node: torch.fx.Node):
 print(node)
 return hook
```

### `torch.ao.quantization.pt2e.graph_utils.get_control_flow_submodules` is no longer public (#141612)
We are planning to make all functions under `torch.ao.quantization.pt2e.graph_utils` private. This update marks `get_control_flow_submodules` as a private API. If you have to or want to continue using `get_control_flow_submodules`, please make a private call by using `_get_control_flow_submodules`.

**Example:**
Version 2.6:
```python
>>> from torch.ao.quantization.pt2e.graph_utils import get_control_flow_submodules
 ```

Version 2.7:
```python
>>> from torch.ao.quantization.pt2e.graph_utils import get_control_flow_submodules
ImportError: cannot import name 'get_control_flow_submodules' from 'torch.ao.quantization.pt2e.graph_utils'
>>> from torch.ao.quantization.pt2e.graph_utils import _get_control_flow_submodules # Note: Use _get_control_flow_submodules for private access
```

---

# Deprecations

### `torch.onnx.dynamo_export` is deprecated (#146425, #146639, #146923)

Users should use the `dynamo=True` option on `torch.onnx.export`.

Version 2.6.0

```python
torch.onnx.dynamo_export(model, *args, **kwargs)
```

Version 2.7.0

```python
torch.onnx.export(model, args, kwargs=kwargs, dynamo=True)
```

### `XNNPACKQuantizer` is deprecated in PyTorch and moved to ExecuTorch, please use it from `executorch.backends.xnnpack.quantizer.xnnpack_quantizer` instead of `torch.ao.quantization.quantizer.xnnpack_quantizer`. (#144940)

`XNNPACKQuantizer` is a quantizer for xnnpack that was added into pytorch/pytorch for initial development. However, as it is not related to our core quantization workflow, we have moved it to ExecuTorch instead. Please use it from `executorch.backends.xnnpack.quantizer.xnnpack_quantizer` instead of `torch.ao.quantization.quantizer.xnnpack_quantizer`.

Version 2.6.0
```python
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
 XNNPACKQuantizer,
 get_symmetric_quantization_config,
)
quantizer = XNNPACKQuantizer().set_global(
 get_symmetric_quantization_config()
)
m = capture_pre_autograd_graph(m, *example_inputs)
m = prepare_pt2e(m, quantizer)
```
Version 2.7.0
```python
# we also updated the export call
from torch.export import export
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
# please get xnnpack quantizer from executorch (https://github.com/pytorch/executorch/)
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
 XNNPACKQuantizer,
 get_symmetric_quantization_config,
)
quantizer = XNNPACKQuantizer().set_global(
 get_symmetric_quantization_config()
)
m = export(m, *example_inputs)
m = prepare_pt2e(m, quantizer)
```

---

# New features

## Release Engineering
- Added support for CUDA 12.8 in CI/CD (#145567, #145789, #145792, #145765, #146019, #146378, #146957, #147037, #146265, #147607, #148000, #149584)
- Added Python 3.13 and 3.13t support in CI/CD (#144698, #143078, #144697, #143074, #141806, #146614)
- Added aarch64 support for pytorch-triton package (#148768, #148705)
- Added support Windows XPU CI/CD (#148755, #147637, #148313, #143185, #148319, #144316, #144644, #144034, #145255)
- Added support for ROCm MI300 CI/CD (#143673, #145504, #146675, #147904, #145398, #145621, #145829, #145790, #144594)
- Added support for [PEP585](https://peps.python.org/pep-0585/), Type Hinting Generics In Standard Collections (#145707, #145177, #145708, #145342, #145101)
- Added Windows Arm64 Nightly Builds (#139760)


## Python Frontend
- Introduce a new `torch.utils.serialization.config` namespace for all serialization related configurations (#143324)
- Add `torch.serialization.config.save.use_pinned_memory_for_d2h` to speed up `torch.save` when passed gpu devices (#143342)
- Add `torch.utils.serialization.config.load.calculate_storage_offsets` to reduce random reads and significantly improve performance for storage with bad random access performance (#143880)
- Add support for `__torch_function__` handler on dtype arguments, similar to subclass objects (#145085)


## C++ Extensions
- Support libtorch-agnostic extensions with stable torch ABI (#148892, #148832, #148124, #149208, #149052)


## Distributed
#### Context Parallel
- We provided a Context Parallel API (#131351) for users to parallelize `torch.nn.functional.scaled_dot_product_attention` over the sequence dimension. We implemented
 Ring Attention (#131351) and an AllGather-based approach (#132820) where the all-gather is issued before the first local SDPA
 and the subsequent local SDPAs will have to wait until the all-gather completes, and offered a user API (#142093) to select the desired approach. The implementation
 currently supports three SDPA kernels: `SDPBackend.FLASH_ATTENTION`, `SDPBackend.EFFICIENT_ATTENTION`, and `SDPBackend.CUDNN_ATTENTION` (#148537). We also
 verified that our Context Parallel implementation is compatible with other parallelisms and `torch.compile`.
#### c10d
- Implemented ncclCommInitRankScalable (merging #136789) (#144794)
#### Distributed Checkpoint (DCP)
- Cache save plans: to mitigate overhead from planning steps (#147116, #147343)
- Build a storage reader/writer to write checkpoints in HF format (#148089)


## CUDA
- Blackwell support added across native kernels, CUDA math libraries, and `torch.compile` (#145270)
- Make `torch.cuda.gds` APIs public (#147120)


## MPS
- Prototype of torch.compile for Metal (#143893)
- Provide Metal kernel authoring via Python (#148972)


## ROCm
- CK Memory-Efficient Attention (attention bias support) (#147778)
- CK Flash Attention Backend (#143695)
- Enhanced Windows support for PyTorch on ROCm (#148563, #144098)
- Support for gfx1102 arch (Navi33) in wheel builds (#147761)
- hipblaslt rowwise f8 gemm (#144432)


## XPU
- Add AOT Inductor support for Intel GPU (#140269, #140664, #149175)
- Support `torch.compile` on Windows Platform for XPU (#147637, #144316, #149511)
- Support SYCL with `torch.utils.cpp_extension` APIs (#132945)
- Enhance Intel GPU performance on PyTorch 2 Export Post Training Quantization (#136753, #135465,#135337, #135189)
- Enable windows Kineto profiler(#148319)
- Enable TF32 support for XPU based on oneDNN backend (#137570)


## torch.compile
#### Dynamo
- Support tracing `contextlib.contextmanager` in Dynamo (#136033)
- `nonstrict_trace` escape hatch to apply non-strict tracing to difficult-to-compile code (#146367)
- Delayed compile for dynamic shapes (#147983)
- Support tracing generators (#141055)
- Whitelist of source files to apply dynamic shapes to (#147979)
- Support tracing `list` subclasses (#146819)
#### Inductor
- Enable non power-of-2 `head_dim` for FlexAttention (#133495).
- Add FlexAttention kernel parameter tuning options: `num_warps` and `num_stages` (#139639).
- Support vectorization for score and mask in FlexAttention CPU (#143638).
- `ConfigFuzzer`: a new debugging tool designed to fuzz Torch compile configurations. Given a test function, it will identify combinations of configs that throw errors during compilation and execution (#139736) (#145565).
- Support fusion of pointwise ops into Template Prologues. `TORCHINDUCTOR_PROLOGUE_FUSION` enables this feature (#147008).
- Add instantiation level for generating configs in the CUTLASS backend. Set `TORCHINDUCTOR_CUTLASS_INSTANTIATION_LEVEL`. Consult config.py for information (#146230).
- Add L2 Swizzle config for CUTLASS backend: `cuda.cutlass_max_profiling_swizzle_options` (#146088).
- Emit a CMakeLists.txt when `package_cpp_only` is specified in AOTI (#143352).
- One Dynamo graph can now map to multiple inductor graphs with different `graph_partition` functions. Set the `graph_partition` in inductor config to enable (#147038).



## Profiler
- Add overload names to profiler (#143114)
- Enable profiling on all threads via `experimentalConfig` (#143659)


## Quantization
- Enables kernel from KleidAI to run model that was quantized such that weights are in int4 (with symmetric quantization either using channel-wise or group-wise, with the group size being a multiple of 32), while at runtime the activations are dynamically quantized from fp32 to int8 and weights are upcast from int4 to int8 so that int8 matrix multiplication is executed. This dynamic quantization of activations and matrix multiplication is performed inside of function `torch.ops.aten._dyn_quant_matmul_4bit`, while the weights, scaled and optional bias are packed in `torch.ops.aten._dyn_quant_pack_4bit_weight`. To use it on your model you can quantize it using the following example that leverages `torchao`:
```python
from torchao.dtypes import PlainLayout
from torchao.experimental.packed_linear_int8_dynamic_activation_intx_weight_layout import (
 PackedLinearInt8DynamicActivationIntxWeightLayout,
)
from torchao.experimental.quant_api import (
 int8_dynamic_activation_intx_weight,
)
from torchao.quantization.granularity import (
 PerGroup,
 PerRow,
)
from torchao.quantization.quant_api import quantize_
from torchao.quantization.quant_primitives import MappingType
my_model = Model()
quantize_(
 my_model,
 int8_dynamic_activation_intx_weight(
 weight_dtype=torch.int4,
 granularity=PerGroup(32), # PerRow() is also supported
 has_weight_zeros=True, # Should be True
 weight_mapping_type=MappingType.SYMMETRIC_NO_CLIPPING_ERR # MappingType.SYMMETRIC can also be used but increases error
 layout=PackedLinearInt8DynamicActivationIntxWeightLayout(target="aten"),
 ),
)
```


## ONNX
#### `torch.onnx.verification.verify_onnx_program` (#148396, #148706, #148730, #148707)

A new verification API `torch.onnx.verification.verify_onnx_program` can now be used to verify numerical accuracy of the exported ONNX model. Users can use the `compare_intermediates` option to identify any operator that causes numerical discrepancies in intermediate tensors. It is possible to use a tool like [model-explorer](https://github.com/justinchuby/model-explorer-onnx) to visualize the verification results.

- Support custom axis name through `dynamic_shapes` (#146321)
- `torch.onnx.export(dynamo=True)` now optimizes the output model by default (#146187)