# pytorch/pytorch v2.9.0

Source: https://github.com/pytorch/pytorch/releases/tag/v2.9.0

*(Features, highlights, backwards-incompatible changes, deprecations, and new features only.)*

# Highlights

 
 
 Unstable (API-Unstable) 
 
 
 Updates to the stable libtorch ABI for third-party C++/CUDA extensions 
 
 
 Symmetric memory that enables easy programming of multi-GPU kernels 
 
 
 The ability to arbitrarily toggle error or resume on graph breaks in torch.compile 
 
 
 Expanded wheel variant support to include ROCm, XPU and CUDA 13 
 
 
 FlexAttention enablement on Intel GPUs 
 
 
 Flash decoding optimization based on FlexAttention on X86 CPU 
 
 
 ARM Platform improvements and optimizations 
 
 
 Enablement of Linux aarch64 binary wheel builds across all supported CUDA versions 
 
 

For more details about these highlighted features, you can look at the [release blogpost](https://pytorch.org/blog/pytorch-2-9/). Below are the full release notes for this release.

---

# Backwards Incompatible Changes


## Min supported Python version is now 3.10 (#162310)

The minimum version of Python required for PyTorch 2.9.0 is 3.10. We also have 3.14 and 3.14t available as preview with this release.


## Undefined behavior when an output of a custom operator shares storage with an input

This is a reminder that outputs of PyTorch custom operators (that are registered using the `torch.library` or `TORCH_LIBRARY` APIs) are not allowed to return Tensors that share storage with input tensors. The violation of this condition leads to undefined behavior: sometimes the result will be correct, sometimes it will be garbage.

After [#163227](https://github.com/pytorch/pytorch/pull/163227), custom operators that violated this condition that previously returned correct results under `torch.compile` may now return silently incorrect results under `torch.compile`. Because this is changing the behavior of undefined behavior, we do not consider this to be a bug, but we are still documenting it in this section as a "potentially unexpected behavior change".

This is one of the conditions checked for by [`torch.library.opcheck`](https://docs.pytorch.org/docs/stable/library.html#testing-custom-ops) and is mentioned in [The Custom Operators Manual](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#bookmark=id.4c0um7xkba6e)

### More details

Outputs of PyTorch custom operators are not allowed to return Tensors that share storage with input tensors

For example, the following two custom operators are not valid custom operators:

```py
@torch.library.custom_op("mylib::foo", mutates_args=())
def foo(x: torch.Tensor) -> torch.Tensor:
 # the result of `foo` must not directly be an input to foo.
 return x

@torch.library.custom_op("mylib::bar", mutates_args=())
def bar(x: torch.Tensor) -> torch.Tensor:
 # the result of bar must not be a view of an input of bar
 return x.view(-1)
```
The easiest workaround is to add an extra `.clone()` to the outputs:
```py
@torch.library.custom_op("mylib::foo", mutates_args=())
def foo(x: torch.Tensor) -> torch.Tensor:
 return x.clone()

@torch.library.custom_op("mylib::bar", mutates_args=())
def bar(x: torch.Tensor) -> torch.Tensor:
 return x.view(-1).clone()
```

A common way to get into this situation is for a user to want to create a custom operator that sometimes mutates the input in-place and sometimes returns a new Tensor, like in the following example.

```py
@torch.library.custom_op("mylib::baz", mutates_args=["x"])
def baz(x: torch.Tensor) -> torch.Tensor:
 if inplace:
 x.sin_()
 return x
 else:
 return x.sin()
```
This dynamism is not supported and leads to undefined behavior. The workaround is to split the custom operator into two custom operators, one that always mutates the input in-place, and another that always returns a new Tensor.
```py
@torch.library.custom_op("mylib::baz_outplace", mutates_args=())
def baz_outplace(x: torch.Tensor) -> torch.Tensor:
 return x.sin()

@torch.library.custom_op("mylib::baz_inplace", mutates_args=["x"])
def baz_inplace(x: torch.Tensor) -> torch.Tensor:
 x.sin_()

def baz(x):
 if inplace:
 baz_inplace(x)
 return x
 else:
 return baz_outplace(x)
```


## Build metal kernels of MacOS-14+ and remove all pre-MacOS-14 specific logic, requires MacOS-14+ going forward (#159733, #159912)

PyTorch MPS is only supported on MacOS-14 or later. If you need to use MPS on MacOS Ventura, please avoid updating to Python-3.9 or above


## Upgrade to DLPack 1.0 (#145000)

This upgrade is doing the same BC-breaking changes as the DLPack release. Objects in `torch.utils.dlpack` have been updated to reflect these changes, such as `DLDeviceType`.

See the PR for details on the exact changes and how to update your code.


## Raise appropriate errors in `torch.cat` (#158249)

`torch.cat` now raises `ValueError`, `IndexError` or `TypeError` where appropriate instead of the generic `RuntimeError`. If you code was catching these errors, you can update to catch the new error type.



## Default to `dynamo=True` for ONNX exporter (#159646, #162726)

Previously `torch.onnx.export(...)` used the legacy TorchScript exporter if no arguments were provied. The ONNX exporter now uses the newer `torch.export.export` pipeline by default (`dynamo=True`). This change improves graph fidelity and future-proofs exports, but may surface graph capture errors that were previously masked or handled differently.

Previously in torch 2.8.0:

```python
# API calls the legacy exporter with dynamo=False
torch.onnx.export(...)
```

Now in torch 2.9.0:

```python
# To preserve the original behavior
torch.onnx.export(..., dynamo=False)

# Export onnx model through torch.export.export
torch.onnx.export(...)
```

Recommendation: first try the new default; only fall back if you hit blocking issues and report them upstream.
Long term solution: fix the root cause instead of relying on fallback or TorchScript exporter.


## Switch off runtime asserts by default in Export in favor of a shape guards function (#160111, #161178, #161794)


To enable runtime asserts, use `export(..., prefer_deferred_runtime_asserts_over_guards=True)`. Also kills the `allow_complex_guards_as_runtime_asserts` flag, merging it into the former option.


Additionally, `exported_program.module()` will generate a call to a `_guards_fn` submodule that will run additional checks on inputs. Users who do not want this behavior can either remove this call in the graph, or do `exported_program.module(check_guards=False)` to avoid the generation.


## Set default opset to 20 in ONNX (#158802)

Opset 20 enables newer operator definitions. If your tooling or downstream runtime only supports opset 18, pin it explicitly. For the latest ONNX operators, you can experiment with opset 23.

Previously in torch 2.8.0:

```python
# opset_version=18
torch.onnx.export(...)
```

Now in torch 2.9.0:

```python
# To preserve the original behavior
torch.onnx.export(..., opset_version=18)

# New: opset_version=20
torch.onnx.export(...)

# Use the latest supported opset: opset_version=23
torch.onnx.export(..., opset_version=23)
```


## Drop `draft_export` in exporter API (#161454, #162225)

Remove implicit draft tracing from the default exporter path, achieving clearer behaviour and faster failures.
The expensive `torch.export.draft_export` diagnostic path is no longer auto-invoked (which could take hours on large models). You can still opt in for deep diagnostics:

Previously in torch 2.8.0:

```bash
# If both torch.export.export(..., strict=False) and
# torch.export.export(..., strict=True) fail to capture
# the model graph, torch.export.draft_export(...) will be triggered,
# and uses real tensor to trace/export the model.
#
# Inside export_to_onnx.py:
# ... torch.onnx.export(..., dynamo=True)
python export_to_onnx.py
```

Now in torch 2.9.0:

```bash
# To trigger torch.export.draft_export once
# torch.export.export strict=False/True both
# fail:

TORCH_ONNX_ENABLE_DRAFT_EXPORT=True python export_to_onnx.py
```


## Remove `torch.onnx.dynamo_export` and the `onnxrt` torch compile backend (#158130, #158258)

`torch.onnx.dynamo_export` is removed. Please use `torch.onnx.export` instead.
The experimental ONNX Runtime compile backend (`torch.compile(backend="onnxrt")`) is no longer supported.


## Remove `torch.onnx.enable_fake_mode` (#161222)

The `dynamo=True` mode uses `FakeTensor`s by default which is memory efficient.


## Some public facing ONNX utility APIs for the TorchScript based exporter are now private (#161323)

Deprecated members in `torch.onnx.verification` are removed. Previously private `torch.onnx.symbolic_opsets*` functions will no longer be accessible. Consider making a copy of the source code if you need to access any private functions for compatibility with the TorchScript based exporter.


## Remove `torch.onnx.symbolic_caffe2` (#157102)

Support for `caffe2` in the ONNX exporter has ended and is removed.


## Remove `/d2implyavx512upperregs` flag that slows build (#159431)

Re-introduced AVX512 optimizations for Windows VS2022 builds, may cause issues with specific versions of VS2022, see #145702


## Add `ScalarType` to shim conversion and `stable::Tensor.scalar_type` (#160557)

Before, user extensions could only in abstract pass around obfuscated dtypes appearing as `int32_ts`. Now, users can confidently use `torch::headeronly::ScalarType` in their extensions for major scalar types. This PR enables ABI stability by adding a translation layer through the shim, so that even if the `ScalarType` enum values change in the future, user extensions need not fear.

This change adds ScalarType support for user extensions and is only narrowly BC breaking for unpopular dtypes: `quint*`s, `qint*`s, `Bits*`, `dummy_uint*`s, `dummy_int*`s, `Float8_e8m0fnu`, and `Float4_e2m1fn_x2` in the use case where an extension retrieves a Tensor dtype of the above and passes it into `aoti_torch_call_dispatcher`.

---

# Deprecations

## Deprecate `pin_memory_device` param in `torch.utils.data.DataLoader` (#158323)

We move enabling `pin_memory` back inside `BaseDataLoaderIter`. This is required for `StatefulDataloader` which leveraged `BaseDataLoaderIter` direclty rather than the `Dataloader` class init


## Deprecate `torch.export.export_for_training` API in favor of equivalent `torch.export.export` API (#158203)

`torch.export.export_for_training` exists because we couldn't migrate internal usages of export to the final IR. Now that we have completed the migration, we deprecated and deleted this API.

---

# New Features

## Python Frontend
- Add utility to get the kernel currently registered on the dispatcher (#158393)
- Extend `__torch_function__` handler to be triggered by elements within a list (#160256)
- Add `torch.hash_tensor` reduction function (#154149)


## FX
- Extend torch function support to ALL arguments instead of just scalar type (but not inside of list, #145089)
- Add `is_fx_symbolic_tracing` flag (#161385)


## Dynamo
- Experimental API for ahead-of-time compiling models in fullgraph mode (#161383)
- Add a hook for recompilations (#157961)
- DynamicInts prototype (#162194)

Introduces an API for annotating dynamic integer inputs & attributes for `torch.compile`, by wrapping plain ints with `DynamicInt()`.
DynamicInt objects also work in eager mode, acting as their underlying values when passed as scalar inputs.

```python
a = DynamicInt(4)
y = a + 2 # DynamicInt(6)
z = torch.ones(a) # torch.ones(4)

fn = torch.compile(torch.ones)
fn(a) # compiled fn takes a dynamic integer input
fn(2) # returns torch.ones(2) without recompiling
```



## Optimizer
- Introduce Muon optimizer to PyTorch (#160213)


## Profiler
- Add GC Events to Python Stack Tracer (#161209)
- Add a custom profiler configuration option (#151656)


## Inductor
- Allow user to pass in custom partitioner function (#157580)


## Export
- Add support for param mutation under inference mode (#159661)


## AOTDispatcher
- Add AOTDispatcher config to set backward autocast behavior (#156356)


## Quantization
- Enable cpu fp8 qlinear and cpu fp8 qconv (#155678, #157076)


## ONNX
- RMS Norm support in opset 23 (#159377)


## C++ Extensions
- Build out a stable set of ATen ops in `torch/csrc/stable/ops.h`: `amax`, `narrow`, `new_empty` + `new_zeros` dtype variant, `pad`, (#159328, #158974, #159508, #161597, #160214)
- Add `torch::stable::Tensor()` default constructor, `is_cpu`, and `get_device_index`(#159507, #160212, #160143)
- Add beginnings of `torch::stable::accelerator` with support for DeviceGuard and Stream (#159679, #160453)
- Start building out `torch/headeronly`: c10 Macros, STD_TORCH_CHECK, ScalarTypes (like BFloat16 and Half, #158035, #158365, #157912, #158377, #159302, #159414, #159412, #159415, #159411, #159911)
- Remove cmake cache and reconfigure again if it is invalid (#156958)
- Cut a version of `TORCH_ERROR_CODE_CHECK` in `headeronly` from AOTI (#159604)
- Remove `wheel` from build requirements (#158027)
- Error when `TORCH_STABLE_ONLY` is defined in `TensorBase.h` (#161658)


## Build Frontend
- Add transpose to `torch/csrc/stable` (#158160)
- Add `zero_()` and `empty_like(t)` to `torch/csrc/stable/ops.h` (#158866)


## Release Engineering
- Add support for CUDA 13.0 in CI/CD builds. Enable CUDA compression mode for binary size reduction for CUDA 13.0 builds (#160956, #161073, #161257, #161663, #161316, #160201, #160770, #161013, #161916, #162268, #162322, #162383, #161833)

- Enable CUDA 12.6, 12.8 and 13.0 support for Linux ARM64 CD builds (#162364, #160720, #159481)

- Add support for Python 3.14 in CI/CD builds (#156889, #157559, #159261, #159869, #160593, #160788, #161255, #159725)

- Enable NVSHMEM integration (#151261, #153010, #154538, #155506, #156685, #158938, #161321, #160778, #159907, #160465)


## CUDA
- Add getter for CUDA graph exec to allow mutation of captured kernel params (#161294)
- Implement support for `cudnn_batch_norm_out` kernel to replace the autogen approach (#123020)


## CPU
- Support GQA for flash attention (#157893)


## MPS
- Partial sparse support for MPS backend (#159729, #160254, #160223, #161846, #162007, #157238)
- Add `avg_pool3d`, `max_unpool1d/2d/3d`, `max_pool3d`, `max_pool3d` bwd pass, and `avg_pool3d` bwd pass for MPS (#158877,#159789, #156467, #157498, #159089)


## ROCm
- OCP Micro-scaling Format (mx-fp8/mx-fp4) Support (#151360)


## XPU
- Enable `FlexAttention` on Intel GPU (#143553)