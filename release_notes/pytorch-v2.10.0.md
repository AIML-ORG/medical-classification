# pytorch/pytorch v2.10.0

Source: https://github.com/pytorch/pytorch/releases/tag/v2.10.0

*(Features, highlights, backwards-incompatible changes, deprecations, and new features only.)*

# Highlights

 
 
 
 Python 3.14 support for torch.compile(). Python 3.14t (freethreaded build) is experimentally supported as well.
 
 
 
 Reduced kernel launch overhead with combo-kernels horizontal fusion in torchinductor 
 
 
 A new varlen_attn() op providing support for ragged and packed sequences 
 
 
 Efficient eigenvalue decompositions with DnXgeev 
 
 
 torch.compile() now respects use_deterministic_mode 
 
 
 DebugMode for tracking dispatched calls and debugging numerical divergence - This makes it simpler to track down subtle numerical bugs. 
 
 
 Intel GPUs support: Expand PyTorch support to the latest Panther Lake on Windows and Linux by enabling FP8 (core ops and scaled matmul) and complex MatMul support, and extending SYCL support in the C++ Extension API for Windows custom ops. 
 
 

For more details about these highlighted features, you can look at the [release blogpost](https://pytorch.org/blog/pytorch-2-10-release-blog/). Below are the full release notes for this release.

---

# Backwards Incompatible Changes

## Dataloader Frontend
- Removed unused `data_source` argument from Sampler ([#163134](https://github.com/pytorch/pytorch/pull/163134)). This is a no-op, unless you have a custom sampler that uses this argument. Please update your custom sampler accordingly.
- Removed deprecated imports for torch.utils.data.datapipes.iter.grouping ([#163438](https://github.com/pytorch/pytorch/pull/163438)). `from torch.utils.data.datapipes.iter.grouping import SHARDING_PRIORITIES, ShardingFilterIterDataPipe` is no longer supported. Please import from `torch.utils.data.datapipes.iter.sharding` instead.


## torch.nn
- Remove Nested Jagged Tensor support from `nn.attention.flex_attention` ([#161734](https://github.com/pytorch/pytorch/pull/161734))


## ONNX
- `fallback=False` is now the default in `torch.onnx.export` ([#162726](https://github.com/pytorch/pytorch/pull/162726))
- The exporter now uses the `dynamo=True` option without fallback. This is the recommended way to use the ONNX exporter. To preserve 2.9 behavior, manually set `fallback=True` in the `torch.onnx.export` call.


## Release Engineering
- Rename pytorch-triton package to triton ([#169888](https://github.com/pytorch/pytorch/pull/169888))

---

# Deprecations

## Distributed
- DeviceMesh
 - Added a warning for slicing flattened dim from root mesh and types for _get_slice_mesh_layout ([#164993](https://github.com/pytorch/pytorch/pull/164993))

We decided to deprecate an existing behavior which goes against the PyTorch design principle (explicit over implicit) for device mesh slicing of flattened dim.

### Version <2.9
```python
import torch
from torch.distributed.device_mesh import

device_type = (
 acc.type
 if (acc := torch.accelerator.current_accelerator(check_available=True))
 else "cpu"
)
mesh_shape = (2, 2, 2)
mesh_3d = init_device_mesh(
 device_type, mesh_shape, mesh_dim_names=("dp", "cp", "tp")
)

mesh_3d["dp", "cp"]._flatten()
mesh_3["dp_cp"] # This comes with no warning
```

### Version >=2.10
```python
import torch
from torch.distributed.device_mesh import

device_type = (
 acc.type
 if (acc := torch.accelerator.current_accelerator(check_available=True))
 else "cpu"
)
mesh_shape = (2, 2, 2)
mesh_3d = init_device_mesh(
 device_type, mesh_shape, mesh_dim_names=("dp", "cp", "tp")
)

mesh_3d["dp", "cp"]._flatten()
mesh_3["dp_cp"] # This will come with a warning because it implicitly change the state of the original mesh. We will eventually remove this behavior in future release. User should do the bookkeeping of flattened mesh explicitly.
```


## Ahead-Of-Time Inductor (AOTI)
- Move `from`/`to` to `torch::stable::detail` ([#164956](https://github.com/pytorch/pytorch/pull/164956))


## JIT
- `torch.jit` is not guaranteed to work in Python 3.14. Deprecation warnings have been added to user-facing `torch.jit` API ([#167669](https://github.com/pytorch/pytorch/pull/167669)).

`torch.jit` should be replaced with `torch.compile` or `torch.export`.


## ONNX
- The `dynamic_axes` option in `torch.onnx.export` is deprecated ([#165769](https://github.com/pytorch/pytorch/pull/165769))

Users should supply the `dynamic_shapes` argument instead. See https://docs.pytorch.org/docs/stable/export.html#expressing-dynamism for more documentation.


## Profiler
- Deprecate `export_memory_timeline` method ([#168036](https://github.com/pytorch/pytorch/pull/168036))

The `export_memory_timeline` method in `torch.profiler` is being deprecated in favor of the newer memory snapshot API (`torch.cuda.memory._record_memory_history` and `torch.cuda.memory._export_memory_snapshot`). This change adds the deprecated decorator from `typing_extensions` and updates the docstring to guide users to the recommended alternative.

---

# New Features

## Autograd
- Allow setting grad_dtype on leaf tensors ([#164751](https://github.com/pytorch/pytorch/pull/164751))
- Add Default Autograd Fallback for PrivateUse1 in PyTorch ([#165315](https://github.com/pytorch/pytorch/pull/165315))
- Add API to annotate disjoint backward for use with `torch.utils.checkpoint.checkpoint` ([#166536](https://github.com/pytorch/pytorch/pull/166536))


## Complex Frontend
- Add `ComplexTensor` subclass ([#167621](https://github.com/pytorch/pytorch/pull/167621))


## Composability
- Support autograd in torch.cond ([#165908](https://github.com/pytorch/pytorch/pull/165908))


## cuDNN
- BFloat16 support added to cuDNN RNN ([#164411](https://github.com/pytorch/pytorch/pull/164411))
- [cuDNN][submodule] Upgrade to cuDNN frontend 1.16.1 (#170591)


## Distributed
- LocalTensor:
 - `LocalTensor` is a powerful debugging and simulation tool in PyTorch's distributed tensor ecosystem. It allows you to simulate distributed tensor computations across multiple SPMD (Single Program, Multiple Data) ranks on a single process. This is incredibly valuable for: 1) debugging distributed code without spinning up multiple processes; 2) understanding DTensor behavior by inspecting per-rank tensor states; 3) testing DTensor operations with uneven sharding across ranks; 4) rapid prototyping of distributed algorithms. Note that LocalTensor is designed for debugging purposes only. It has significant overhead and is not suitable for production distributed training.
 - `LocalTensor` is a `torch.Tensor` subclass that internally holds a mapping from rank IDs to local tensor shards. When you perform a PyTorch operation on a `LocalTensor`, the operation is applied independently to each local shard, mimicking distributed computation (`LocalTensor` simulates collective operations locally without actual network communication.). `LocalTensorMode` is the context manager that enables `LocalTensor` dispatch. It intercepts PyTorch operations and routes them appropriately. The `@maybe_run_for_local_tensor` decorator is essential for handling rank-specific logic when implementing distributed code.
 - To get started with `LocalTensor`, users import from `torch.distributed._local_tensor`, initialize a fake process group, and wrap their distributed code in a `LocalTensorMode` context. Within this context, DTensor operations automatically produce LocalTensors.
 - PRs: ([#164537](https://github.com/pytorch/pytorch/pull/164537), [#166595](https://github.com/pytorch/pytorch/pull/166595), [#168110](https://github.com/pytorch/pytorch/pull/168110),[#168314](https://github.com/pytorch/pytorch/pull/168314),[#169088](https://github.com/pytorch/pytorch/pull/169088),[#169734](https://github.com/pytorch/pytorch/pull/169734))

- c10d:
 - New `shrink_group` implementation to expose `ncclCommShrink` API ([#164518](https://github.com/pytorch/pytorch/pull/164518))


## Dynamo
- `torch.compile` now fully works in Python 3.14 ([#167384](https://github.com/pytorch/pytorch/pull/167384))
- Add option to error or disable applying side effects ([#167239](https://github.com/pytorch/pytorch/pull/167239))
- Config flag (`skip_fwd_side_effects_in_bwd_under_checkpoint`) to allow eager and compile activation-checkpointing divergence for side-effects ([#165775](https://github.com/pytorch/pytorch/pull/165775))
- `torch._higher_order_ops.print` for enabling printing without graph breaks or reordering ([#167571](https://github.com/pytorch/pytorch/pull/167571))


## FX
- Added node metadata annotation API
- Disable preservation of node metadata when `enable=False` ([#164772](https://github.com/pytorch/pytorch/pull/164772))
- Annotation should be mapped across submod ([#165202](https://github.com/pytorch/pytorch/pull/165202))
- Annotate bw nodes before eliminate dead code ([#165782](https://github.com/pytorch/pytorch/pull/165782))
- Add logging for debugging annotation ([#165797](https://github.com/pytorch/pytorch/pull/165797))
- Override metadata on regenerated node in functional mode ([#166200](https://github.com/pytorch/pytorch/pull/166200))
- Skip copying custom meta for gradient accumulation nodes; tag with is_gradient_acc=True ([#167572](https://github.com/pytorch/pytorch/pull/167572))
- Add metadata hook for all nodes created in runtime_assert pass ([#169497](https://github.com/pytorch/pytorch/pull/169497))
- Update `gm.print_readable` to include Annotation ([#165397](https://github.com/pytorch/pytorch/pull/165397))
- Add annotation to assertion nodes in export ([#167171](https://github.com/pytorch/pytorch/pull/167171))

- Add debug mode to print meta in fx graphs ([#165874](https://github.com/pytorch/pytorch/pull/165874))


## Inductor
- Add experimental Pallas TorchInductor backend. ([#166822](https://github.com/pytorch/pytorch/pull/166822))
- Add Pallas TPU backend support. ([#167774](https://github.com/pytorch/pytorch/pull/167774))
- Add Flash Attention support to FlexAttention. ([#161118](https://github.com/pytorch/pytorch/pull/161118))
- Add deterministic mode for Inductor compilation. ([#163589](https://github.com/pytorch/pytorch/pull/163589)) ([#165950](https://github.com/pytorch/pytorch/pull/165950)) ([#164532](https://github.com/pytorch/pytorch/pull/164532))
- Enable custom op autotune decompositions and parameter tuning. ([#164212](https://github.com/pytorch/pytorch/pull/164212)) ([#167193](https://github.com/pytorch/pytorch/pull/167193))
- Expose `torch.compiler.config.force_disable_caches` as a public API. ([#166699](https://github.com/pytorch/pytorch/pull/166699))
- Add HOP for additional control dependencies to enforce explicit scheduling. ([#164568](https://github.com/pytorch/pytorch/pull/164568))
- Add Inductor Lite Mode ([#167115](https://github.com/pytorch/pytorch/pull/167115))
- Add distributed autotuning support ([#163369](https://github.com/pytorch/pytorch/pull/163369))
- Add Native matmul support to inductor ([#157743](https://github.com/pytorch/pytorch/pull/157743))


## Ahead-Of-Time Inductor (AOTI)
- Integrate AOTI as a backend. ([#167338](https://github.com/pytorch/pytorch/pull/167338))
- Add AOTI mingw cross compilation for Windows. ([#163188](https://github.com/pytorch/pytorch/pull/163188))


## MPS
- MPS sparse backend is functional
([#162349](https://github.com/pytorch/pytorch/pull/162349), [#162349](https://github.com/pytorch/pytorch/pull/162349), [#162007](https://github.com/pytorch/pytorch/pull/162007), [#162910](https://github.com/pytorch/pytorch/pull/162910), [#162885](https://github.com/pytorch/pytorch/pull/162885), [#163011](https://github.com/pytorch/pytorch/pull/163011), [#163694](https://github.com/pytorch/pytorch/pull/163694), [#164961](https://github.com/pytorch/pytorch/pull/164961), [#165102](https://github.com/pytorch/pytorch/pull/165102), [#166708](https://github.com/pytorch/pytorch/pull/166708), [#166711](https://github.com/pytorch/pytorch/pull/166711), [#167013](https://github.com/pytorch/pytorch/pull/167013), [#169125](https://github.com/pytorch/pytorch/pull/169125), [#165232](https://github.com/pytorch/pytorch/pull/165232), [#166708](https://github.com/pytorch/pytorch/pull/166708), [#168154](https://github.com/pytorch/pytorch/pull/168154), [#169368](https://github.com/pytorch/pytorch/pull/169368), [#167908](https://github.com/pytorch/pytorch/pull/167908), [#168112](https://github.com/pytorch/pytorch/pull/168112))


## torch.nn
- Add `nn.functional.scaled_mm` ([#164142](https://github.com/pytorch/pytorch/pull/164142))
- Add `nn.functional.scaled_grouped_mm` ([#165154](https://github.com/pytorch/pytorch/pull/165154))
- Add `nn.attention.varlen_attn` ([#164502](https://github.com/pytorch/pytorch/pull/164502), [#164504](https://github.com/pytorch/pytorch/pull/164504))
- Add `nn.functional.grouped_mm` ([#168298](https://github.com/pytorch/pytorch/pull/168298))


## ONNX
- A new testing module `torch.onnx.testing` with a testing utility `assert_onnx_program` ([#162495](https://github.com/pytorch/pytorch/pull/162495))


## Profiler
- Add scope for `RecordFunctionFast` ([#162661](https://github.com/pytorch/pytorch/pull/162661))


## Quantization
- Add `_scaled_mm_v2` API ([#164141](https://github.com/pytorch/pytorch/pull/164141))
- Add `scaled_grouped_mm_v2` and python API ([#165154](https://github.com/pytorch/pytorch/pull/165154))
- Add `embedding_bag_byte_prepack_with_rowwise_min_max` and `embedding_bag_{2/4}bit_prepack_with_rowwise_min_max` ([#162924](https://github.com/pytorch/pytorch/pull/162924))

- Add `MXFP4` support for `_scaled_grouped_mm_v2` via. FBGEMM kernels ([#166530](https://github.com/pytorch/pytorch/pull/166530))


## Release Engineering
- Enabled auto-revert on PyTorch CI ([#163858](https://github.com/pytorch/pytorch/pull/163858), [#164911](https://github.com/pytorch/pytorch/pull/164911), [#165459](https://github.com/pytorch/pytorch/pull/165459))
- Add PEP 517 compliant Python source distribution package to release process ([#157815](https://github.com/pytorch/pytorch/pull/157815))
- Add Pallas CI testing infrastructure with CPU and GPU test ([#167143](https://github.com/pytorch/pytorch/pull/167143), [#167428](https://github.com/pytorch/pytorch/pull/167428), [#169687](https://github.com/pytorch/pytorch/pull/169687), [#169494](https://github.com/pytorch/pytorch/pull/169494), [#169802](https://github.com/pytorch/pytorch/pull/169802))


## ROCm
- Enable grouped GEMM via regular GEMM fallback ([#162419](https://github.com/pytorch/pytorch/pull/162419))
- Enable grouped GEMM via CK ([#166334](https://github.com/pytorch/pytorch/pull/166334), [#167403](https://github.com/pytorch/pytorch/pull/167403))
- Enable ATen GEMM overload for FP32 output from FP16/BF16 inputs ([#162600](https://github.com/pytorch/pytorch/pull/162600))
- Support torch.cuda._compile_kernel ([#162510](https://github.com/pytorch/pytorch/pull/162510))
- Enhanced Windows support
- load_inline ([#162577](https://github.com/pytorch/pytorch/pull/162577))
- Enable AOTriton runtime compile ([#165538](https://github.com/pytorch/pytorch/pull/165538))
- AOTriton scaled_dot_product_attention ([#162330](https://github.com/pytorch/pytorch/pull/162330))
- Add gfx1150 gfx1151 to hipblaslt-supported GEMM lists ([#164744](https://github.com/pytorch/pytorch/pull/164744))
- Add scaled_mm v2 support. ([#165528](https://github.com/pytorch/pytorch/pull/165528))
- Add torch.version.rocm, distinct from torch.version.hip ([#168097](https://github.com/pytorch/pytorch/pull/168097))


## XPU
- Support ATen operators `scaled_mm` and `scaled_mm_v2` for Intel GPU ([#166056](https://github.com/pytorch/pytorch/pull/166056))
- Support ATen operator `_weight_int8pack_mm` for Intel GPU ([#160938](https://github.com/pytorch/pytorch/pull/160938))
- Extend SYCL support in PyTorch CPP Extension API to allow users to implement new custom operators on Windows ([#162579](https://github.com/pytorch/pytorch/pull/162579))
- Add API `torch.xpu.get_per_process_memory_fraction` for Intel GPU ([#165511](https://github.com/pytorch/pytorch/pull/165511))
- Add API `torch.xpu.set_per_process_memory_fraction` for Intel GPU ([#165510](https://github.com/pytorch/pytorch/pull/165510))
- Add API `torch.xpu.is_tf32_supported` for Intel GPU ([#163141](https://github.com/pytorch/pytorch/pull/163141))
- Add API `torch.xpu.can_device_access_peer` for Intel GPU ([#162705](https://github.com/pytorch/pytorch/pull/162705))
- Add API `torch.accelerator.get_memory_info` for Intel GPU ([#162564](https://github.com/pytorch/pytorch/pull/162564))