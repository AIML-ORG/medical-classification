# pytorch/pytorch v2.4.0

Source: https://github.com/pytorch/pytorch/releases/tag/v2.4.0

*(Features, highlights, backwards-incompatible changes, deprecations, and new features only.)*

## Highlights

We are excited to announce the release of PyTorch® 2.4!
PyTorch 2.4 adds support for the latest version of Python (3.12) for `torch.compile`.
AOTInductor freezing gives developers running AOTInductor more performance based optimizations by allowing the
serialization of MKLDNN weights. As well, a new default TCPStore server backend utilizing `libuv` has been introduced
which should significantly reduce initialization times for users running large-scale jobs.
Finally, a new Python Custom Operator API makes it easier than before to integrate custom kernels
into PyTorch, especially for `torch.compile`.

This release is composed of 3661 commits and 475 contributors since PyTorch 2.3. We want to sincerely thank our 
dedicated community for your contributions. As always, we encourage you to try these out and report any issues as we 
improve 2.4. More information about how to get started with the PyTorch 2-series can be found at our 
[Getting Started](https://pytorch.org/get-started/pytorch-2.0/) page.


 
 
 Beta
 
 Prototype
 
 Performance Improvements
 
 
 
 Python 3.12 support for torch.compile
 
 FSDP2: DTensor-based per-parameter-sharding FSDP
 
 torch.compile optimizations for AWS Graviton (aarch64-linux) processors
 
 
 
 AOTInductor Freezing for CPU
 
 torch.distributed.pipelining, simplified pipeline parallelism
 
 BF16 symbolic shape optimization in TorchInductor
 
 
 
 New Higher-level Python Custom Operator API
 
 Intel GPU is available through source build
 
 Performance optimizations for GenAI projects utilizing CPU devices
 
 
 
 Switching TCPStore’s default server backend to libuv
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

*To see a full list of public feature submissions click [here](https://docs.google.com/spreadsheets/d/1TzGkWuUMF1yTe88adz1dt2mzbIsZLd3PBasy588VWgk/edit?usp=sharing).

---

## Deprecations

### Python frontend

- User warning when using `torch.load` with default `weights_only=False` value (#129239, #129396, #129509).
 A warning is now raised if the weights_only value is not specified during a call to torch.load, encouraging users to
 adopt the safest practice when loading weights.
- Deprecate device-specific autocast API (#126062)
 All the autocast APIs are unified under torch.amp and it can be used as a drop-in replacement for torch.{device}.amp APIs
 (passing a device argument where applicable)..
- Export torch.newaxis=None for Python Array API/Numpy consistency (#125026)

### Composability

- Deprecate calling FakeTensor.data_ptr in eager-mode. FakeTensors are tensors without a valid data pointer, so in
 general their data pointer is not safe to access. This makes it easier for `torch.compile` to provide a nice error
 message when tracing custom ops into a graph that are not written in a PT2-friendly way (because, for example, they
 try to directly access a tensor’s data pointer from a region of code being traced). More details on integrating custom
 ops with `torch.compile` can be found [here](https://dev-discuss.pytorch.org/t/guide-getting-c-custom-ops-to-work-with-torch-compile/1737) (#123292)
- Dynamic shapes:
 - SymInt-ify mem-efficient attention forward op signature (#125418)
 - Don't call item() into torch.scalar_tensor uselessly (#125373)
 - Fix scalar type for constraint_range to Long (#121752)
 - Guard oblivious on meta registrations (#122216), vector_norm (#126772), and unbind (#124959)
 - Make expected stride test in torch._prims_common size oblivious (#122370)
 - Use torch._check for safety assert in _reshape_view_helper (#125187)
 - Add a code comment about torch._check_is_size in tensor_split (#125292)
 - Make min(stride, strides[idx]) in collapse_view_helper size oblivious (#125301)
 - Don't short circuit if shape is same (#125188)

### CPP

- Refactor autocast C++ APIs to be device-agnostic (#124359)

### Release Engineering

- Remove of QNNPACK third-party module (#126941)

### Optim

- Deprecate LRScheduler.print_lr (#126105)

### nn

- `torch.nn.hardtahn` allowed `min_val` to be greater than max_val (#121627)

### Distributed

- Distributed Checkpointing (DCP)
 Deprecated submodules feature for distributed_state_dict (#127793)
 In 2.3 and before, users can do:
 ```python
 model = AnyModel(device=torch.device("cuda"))
 model_state_dict = get_model_state_dict(model)
 set_model_state_dict(
 model,
 model_state_dict=new_model_state_dict,
 options=StateDictOptions(strict=False),
 )

 # Below way of calling API is also legit
 model_state_dict2 = get_model_state_dict(model, submodules={model.submodule})
 set_model_state_dict(
 model,
 model_state_dict={model.submodule: new_submodel_state_dict},
 options=StateDictOptions(strict=False),
 )
 ```
 But from 2.4 forward, if users call `get_model_state_dict` or `set_model_state_dict` with a submodule path or
 state_dict, users will see a warning about the feature. To achieve the same functionality, users can manually
 filter out the `state_dict` returned from `get_state_dict` API and preprocess the model_state_dict before
 calling `set_state_dict` API:
 ```python
 model = AnyModel(device=torch.device("cuda"))
 model_state_dict = get_model_state_dict(model)
 set_model_state_dict(
 model,
 model_state_dict=new_model_state_dict,
 options=StateDictOptions(strict=False),
 )
 # Deprecating warnings thrown for the below way of calling API
 model_state_dict2 = get_model_state_dict(model, submodules={model.submodule})
 set_model_state_dict(
 model,
 model_state_dict={model.submodule: new_submodel_state_dict},
 options=StateDictOptions(strict=False),
 )
 ```
- FullyShardedDataParallel (FSDP)
 Deprecate FSDP.state_dict_type and redirect users to distributed_state_dict (#127794)
 In 2.3 and before, users can do:
 ```python
 model = AnyModel(device=torch.device("cuda"))
 fsdp_model = FSDP(model)
 # Users can do both ways below
 get_model_state_dict(model)
 with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
 fsdp_model.state_dict()
 ```
 But from 2.4 forward, if users call `state_dict` or set `state_dict` with the FSDP.state_dict_type, users will see warnings. And the recommended solution now is to use `get_model_state_dict` and `set_model_state_dict` directly:
 ```python
 model = AnyModel(device=torch.device("cuda"))
 fsdp_model = FSDP(model)

 get_model_state_dict(model)
 # Deprecating warnings thrown for the below way of calling API
 with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
 fsdp_model.state_dict()
 ```

### Profiler

- Remove FlameGraph usage steps from export_stacks docstring (#123102)
 The export_stacks API will continue to work as before, however we’ve removed the docstring to use FrameGraph.
 PyTorch doesn’t own FrameGraph, and cannot guarantee that it functions properly.

### Quantization

- Remove deprecated `torch._aminmax` operator (#125995).
 `torch._aminmax` -> `torch.aminmax` instead

### Export

- Start deprecation of capture_pre_autograd_graph (#125848, #126403)

### XPU

- Refactor autocast C++ APIs to be device-agnostic(#124359)
 `at::autocast::get_autocast_gpu_dtype()` -> `at::autocast::get_autocast_dtype(at::kCUDA)`
 `at::autocast::get_autocast_cpu_dtype()` -> `at::autocast::get_autocast_dtype(at::kCPU)`
- Refactor autocast Python APIs(#124479)
 `torch.get_autocast_gpu_dtype()` -> `torch.get_autocast_dtype(“cuda”)`,
 `torch.set_autocast_gpu_dtype(dtype)` -> `torch.set_autocast_dtype(“cuda”, dtype)`,
 `torch.is_autocast_enabled() ` -> `torch.is_autocast_enabled(“cuda”)`,
 `torch.set_autocast_enabled(enabled)` -> `torch.set_autocast_enabled(”cuda”, enabled)`,
 `torch.get_autocast_cpu_dtype()` -> `torch.get_autocast_dtype(“cpu”)`
- Make torch.amp.autocast more generic (#125103)
 `torch.cuda.amp.autocast(args…) ` -> `torch.amp.autocast(“cuda”,args…)`,
 `torch.cpu.amp.autocast(args…) ` -> `torch.amp.autocast(“cpu”, args…)`,
- Deprecate device-specific GradScaler autocast API(#126527)
 `torch.cuda.amp.GradScaler(args…) ` -> `torch.amp.GradScaler(“cuda”, args…)`,
 `torch.cuda.amp.GradScaler(args…) ` -> `torch.amp.GradScaler(“cpu”, args…)`,
- Generalize custom_fwd&custom_bwd to be device-agnostic (#126531)
 `torch.cuda.amp.custom_fwd(args…) ` -> `torch.amp.custom_fwd(args…, device_type=’cuda’)`,
### ONNX

- Remove more caffe2 files (#126628)

---

## New Features

### Python frontend

- Add
 - support for unsigned int sizes for torch.unique (#123643)
 - torch.OutOfMemoryError to signify out of memory error from any device (#121702)
 - new device-agnostic API for autocast in torch.amp.* (#124938)
 - new device-agnostic API for Stream/Event in torch.{Stream,Event} (#125757)
 - channels last support to max, average and adaptive pooling functions (#116305)
 - torch.serialization.add_safe_globals that allows users to allowlist classes for weights_only
 load (#124331, #124330, #127808)
 - pickling support for torch.Generator (#126271)
 - torch.utils.module_tracker to track position within torch.nn.Module hierarchy (#125352)

### Composability

- Add
 - OpOverload.redispatch; use it in new custom ops API (#124089)
 - mutated_args field to custom_op (#123129)
 - new Python Custom Operators API
 - register_autograd to register backward formulas for custom ops (#123110)
 - torch.library.opcheck (#124496), torch.library.register_autograd (#124071), torch.library.register_kernel (#124299)
- Blanket ban kwarg-only Tensors (#124805)
- Change register_autograd to reflect ordering of setup_context and backward (#124403)
- Ensure torch.library doctests runs under xdoctest (#123282)
- Fix torch.library.register_fake's module reporting (#125037)
- New Custom Ops Documentation landing page (#127400)
- Refresh OpOverloadPacket if a new OpOverload gets added (#126863, #128000)
- Rename
 - impl_abstract to register_fake, part 1/2 (#123937)
 - register_impl to register_kernel (#124200)
- Schema inference now includes default values (#123453)
- Stop requiring a pystub for register_fake by default (#124064)
- Support TensorList inputs/outputs (#123615)
- Update the functionalization error message (#123261)
- add ability to provide manual schema (#124180)
- fix schema inference for kwarg-only args (#124637)
- mutated_args -> mutates_args (#123437)
- register_autograd supports non-tensor kwargonly-args (#124806)
- set some tags when constructing the op (#124414)
- setup_context fills in default values (#124852)
- torch.library.register_fake accepts more types (#124066)
- use new python custom ops API on prims ops (#124665)

### Optim

- Enable `torch.compile` support for LRScheduler with Tensor LRs (#123751, #123752, #123753, #127190)

### nn frontend

- Add RMSNorm module (#121364)

### linalg

- Implement svd_lowrank and pca_lowrank for complex numbers (#125580)
- Extend `preferred_backend` on ROCm backend.
- Add cuBLASLt `gemm` implementation (#122106)

### Distributed

#### c10d

- Implemented IntraNodeComm primitives for `allgather_matmul` (#118038)
- Add first differentiable collective `all_to_all_single_grad` (#123599)
- Add P2P versions of `send/recv_object_list` operations (#124379)
- Add a new Collectives API for doing distributed collectives operations in the Elastic
 store with more performant and debuggable primitives (#126695)

#### FullyShardedDataParallel v2 (FSDP2)

- FSDP2 is a new fully sharded data parallel implementation that uses DTensor-based dim-0 per-parameter
 sharding for improved flexibility (e.g. mixed-dtype all-gather, no constraints on requires_grad) without
 significant cost to performance.
 See the [document](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md) for more details and a
 comparison with FSDP1 (#122888, #122907, #123142, #123362, #123491, #123857, #119302, #122908, #123953, #120952, #123988, #124293, #124318, #124319, #120256, #124513, #124955, #125191, #125269, #125394, #126070, #126267, #126305, #126166, #127585, #127776, #127832, #128138, #128117, #128242)

#### Pipelining

- PyTorch Distributed pipeline parallelism APIs were upstreamed from the
 [PiPPy project](https://github.com/pytorch/PiPPy) and are available as a prototype release in
 PyTorch 2.4.
 The package is under [torch.distributed.pipelining](https://github.com/pytorch/pytorch/tree/main/torch/distributed/pipelining)
 and consists of two parts: a splitting frontend and a distributed runtime.
 The splitting frontend takes your model code as-is, splits it up into “model partitions”, and captures the data-flow relationship.
 The distributed runtime executes the pipeline stages on different devices in parallel, handling things like micro-batch splitting,
 scheduling, communication, and gradient propagation.
 For more information please check out the
 [documentation](https://pytorch.org/docs/main/distributed.pipelining.html) and
 [tutorial](https://pytorch.org/tutorials/intermediate/pipelining_tutorial.html) (#126322, #124776, #125273, #125729, #125975, #126123, #126419, #126539, #126582, #126732, #126653, #127418, #127084, #127673, #127332, #127946, #128157, #128163, #127796, #128201, #128228, #128240, #128236, #128273, #128279, #128276, #128278, #127066)

### Profiler

- Add profiler support for `PrivateUse1` (#124818)

### Dynamo

- `torch.compile` is compatible with Python 3.12.
- Guarding on nn modules attributes (#125202) - TorchDynamo guards on nn module attributes. This was a frequently raised
 issue in the past (examples (#111785, #120248, #120958, #117758, #124357, #124717, #124817)).
 This increases TorchDynamo soundness with minimal perf impact.
- Hardened the recently introduced tracing rules infrastructure. This allows `torch.compile` users to easily control TorchDynamo tracing of PyTorch internal code.
- Extended `torch.compile` support for RAdam and Adamax optimizer. Compiler optimizers now demonstrate SOTA performance.
- Experimental feature - We introduced a new experimental flag torch._dynamo.config.inline_inbuilt_nn_modules to enable `torch.compile` to reuse compiled
 artifacts on repeated blocks in the models. This gives another point in the tradeoff space of compilation time and performance speedup.
 By moving `torch.compile` from full model to a repeated block (e.g. moving `torch.compile` from full LLM to a repeated Transformer block),
 we can now achieve faster compilation time with some performance dip compared to full model.
 We plan to make this flag default to True in the 2.5 release.

### Export

- Introduce ShapesCollection, a dynamic shapes builder API (#124898)

### Inductor

- Add higher order associative scan operator (#119430)

### jit

- Add aten::sort.any op for sorting lists of arbitrary elements (#123982)

### MPS

- Conform torch.mps to device module interface (#124676)

### XPU

- Inductor Intel GPU backend (#121895)
- a new autocast API torch.amp.is_autocast_available(#124938)
- attributes to xpu device prop (#121898)
- XPU implementation for PyTorch ATen operators (#120891)
- generic stream/event on XPU backend (#125751)
- gpu trace on XPU (#121795)
- Switch to torch.float16 on XPU AMP mode (#127741)

### ONNX

- quantized layer norm op to opset 17 (#127640)
- symbolic_opset19.py and symbolic_opset20.py to support opset 19/20, extend opset 18 support (#118828)
- Support for Some Bitwise Ops in Onnx Exporter (#126229)
- Allow ONNX models without parameters (#121904)
- Integrate onnxscript optimizer (#123379)

### Vulkan

- quantized transposed 2D convolutions (#120151, #122547)
- the quantized ReLU operator (#123004)