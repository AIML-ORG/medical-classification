# pytorch/ao v0.9.0

Source: https://github.com/pytorch/ao/releases/tag/v0.9.0

*(Features, highlights, backwards-incompatible changes, deprecations, and new features only.)*

# Highlights

We are excited to announce the 0.9.0 release of torchao! This release moves a number of sparsity techniques out of prototype, a significant overhaul of the quantize_ api, a new cutlass kernel for 4 bit dynamic quantization and more!

### Block Sparsity promoted out of prototype
We’ve promoted block sparsity out of torchao.prototype and made several performance improvements. 
You can accelerate your models with block sparsity as follows:

```python
from torchao.sparsity import sparsify, block_sparse_weight
sparsify_(model, block_sparse_weight(blocksize=64))
```

##### Blocksparse Benchmarks

| **Technique** |**Decode (tok/s)**| **Model Size (GB)** | 
|------------------------------|------------------|---------------------|
| baseline | 134.40 | 15.01 |
| 2:4 sparse | 163.13 | 10.08 |
| bsr-0.8-32 | 210.91 | 6.01 |
| bsr-0.8-64 | 222.43 | 6.00 |
| bsr-0.9-32 | 255.19 | 4.88 |
| bsr-0.9-64 | 262.94 | 4.88 |
| 2:4 sparse + int4wo (marlin) | 255.21 | 3.89 |

Block Sparsity technique names (bsr) indicate sparsity fraction and blocksize.

These numbers were generated on H100 using torchao/_models/llama/generate.py on the Meta-Llama-3.1-8B model. You can reproduce these numbers using this [script](https://gist.github.com/HDCharles/6e782c33d5aac24b36fa81d9e3bd5f5c)


## BC Breaking

### TorchAO M1 Binaries currently not working

W've identified that the binaries are broken on M1 and have been since v0.8.0 though they were working in v0.7.0. We're working on a fix for this, details and discussion can be found [here](https://github.com/pytorch/ao/issues/1796).


#### quantize_ configuration callables -> configs (https://github.com/pytorch/ao/pull/1595, https://github.com/pytorch/ao/pull/1694, https://github.com/pytorch/ao/pull/1696, https://github.com/pytorch/ao/pull/1697) 

We are migrating the way `quantize_` workflows are configured from callables (tensor subclass inserters) to direct configuration (config objects). Motivation: align with the rest of the ecosystem, enable inspection of configs after instantiation, remove a common source of confusion.

**What is changing:**

Specifically, here is how the signature of `quantize_`'s second argument will change:

```python
#
# torchao v0.8.0 and before
#
def quantize(
 model: torch.nn.Module,
 apply_tensor_subclass: Callable[[torch.nn.Module], torch.nn.Module],
 ...,
): ...

#
# torchao v0.9.0
#
def quantize(
 model: torch.nn.Module,
 config: Union[AOBaseConfig, Callable[[torch.nn.Module], torch.nn.Module]],
 ...,
): ...

#
# torchao v0.10.0 or later (exact version TBD)
#
def quantize(
 model: torch.nn.Module,
 config: AOBaseConfig,
 ...,
): ...
```

1. the name of the second argument to `quantize_` changed from `apply_tensor_subclass` to `config`. Since the vast majority of callsites today are passing in configuration with a positional argument, this change should not affect most people.
2. the type of the second argument to `quantize_` will change from `Callable[[torch.nn.Module], torch.nn.Module]` to `config: AOBaseConfig`, following a deprecation process detailed below.
3. for individual workflows, the user facing API name changed from snake case (`int8_weight_only`) to camel case (`Int8WeightOnlyConfig`). All argument names for each config are kept as-is. We will keep the old snake case names (`int8_weight_only`) around and alias them to the new names (`int8_weight_only = Int8WeightOnlyConfig`), to avoid breaking callsites. We plan to keep the old names forever. Here are all the workflow config name changes:

| old name (will keep working) | new name (recommended) |
| --- | --- |
| `int4_weight_only` | `Int4WeightOnlyConfig` |
| `float8_dynamic_activation_float8_weight` | `Float8DynamicActivationFloat8WeightConfig`|
| `float8_static_activation_float8_weight` | `Float8StaticActivationFloat8WeightConfig` |
| `float8_weight_only` | `Float8WeightOnlyConfig` |
| `fpx_weight_only` | `FPXWeightOnlyConfig` |
| `gemlite_uintx_weight_only` | `GemliteUIntXWeightOnlyConfig` |
| `int4_dynamic_activation_int4_weight` | `Int4DynamicActivationInt4WeightConfig` |
| `int8_dynamic_activation_int4_weight` | `Int8DynamicActivationInt4WeightConfig` |
| `int8_dynamic_activation_int8_semi_sparse_weight` | n/a (deprecated) |
| `int8_dynamic_activation_int8_weight` | `Int8DynamicActivationInt8WeightConfig` |
| `int8_weight_only` | `Int8WeightOnlyConfig` |
| `uintx_weight_only` | `UIntXWeightOnlyConfig` |

Configuration for prototype workflows using `quantize_` will be migrated at a later time.

**How these changes can affect you:**
1. If you are a user of existing `quantize_` API workflows and are passing in config by a positional argument (`quantize_(model, int8_weight_only(group_size=128))`), **you are not affected**. This positional syntax will keep working going forward. You are encouraged to migrate your callsite to the new config name (`quantize_(model, Int8WeightOnlyConfig(group_size=128))` though the old names will continue to work indefinitely.
2. If you are a user of existing `quantize_` API workflows and are passing in config by a keyword argument (`quantize_(model, tensor_subclass_inserter=int8_weight_only(group_size=128))`), your callsite will break. You will need to change your callsite to `quantize_(model, config=int8_weight_only(group_size=128))`. We don't expect many people to be in this bucket.
3. If you are a developer writing new workflows for the `quantize_` API, you will need to use the new configuration system. Please see https://github.com/pytorch/ao/issues/1690 for details.
4. If you are a user of `sparsify_`, you are not affected for now and a similar change will happen in a future version of torchao.

This migration will be a two step process:
* in torchao v0.9.0, we will enable the new syntax while starting the deprecation process for the old syntax.
* in torchao v.0.10.0 or later, we will remove the old syntax

Please see https://github.com/pytorch/ao/issues/1690 for more details.


### Block Sparsity imports after moved out of prototype (https://github.com/pytorch/ao/pull/1734)

Before:

```python
from torchao.prototype.sparsity.superblock.blocksparse import block_sparse_weight
```

After:
```python
from torchao.sparsity import block_sparse_weight
```



## Deprecations

#### deprecation of the `set_inductor_config` argument of `quantize_` (https://github.com/pytorch/ao/pull/1716)

We are migrating the `set_inductor_config` argument of `quantize_` to individual workflows. Motivation:
1. this functionality was intended for inference, and we don't want to expose it to future training workflows that we plan to add to `quantize_`.
2. higher level, this flag couples torchao workflows with torch.compile, which is not ideal. We would rather keep these systems decoupled at the `quantize_` API level, with individual workflows opting in as needed.

##### Impact on users

* for torchao v0.9.0:: if you are passing in `set_inductor_config` to `quantize_`, your callsite will keep working with a deprecation warning. We recommend that you migrate this option to your individual workflow.
* for a future version of torchao: the `set_inductor_config` argument will be removed from `quantize_`.

##### API changes

```python
# torchao v0.8.x
def quantize_(
 ...,
 set_inductor_config: bool = True,
 ...,
): ...

# torchao v.0.9.0
def quantize_(
 ...,
 set_inductor_config: Optional[bool] = None,
 ...,
):
 # if set_inductor_config != None, throw a deprecation warning
 # if set_inductor_config == None, set it to True to stay consistent with old behavior

# torchao v TBD (a future release)
def quantize_(
 ...,
):
 # set_inductor_config is removed from quantize_ and moved to relevant individual workflows
```

Please see https://github.com/pytorch/ao/issues/1715 for more details.

#### Deprecation warning for float8 training delayed and static scaling (https://github.com/pytorch/ao/pull/1681, https://github.com/pytorch/ao/issues/1680)
We plan to deprecate delayed and static scaling from torchao.float8 training codebase due to lack of real world use cases for delayed/static scaling (dynamic scaling is required for higher accuracy) and 
complexity tax for supporting these features.
* for torchao v0.9.0: add deprecation warning for delayed and static scaling
* for torchao v0.10.0: deprecate delayed and static scaling



## New Features

#### Supermask for improving accuracy for sparse models (https://github.com/pytorch/ao/pull/1729)

Supermask (https://pytorch.org/blog/speeding-up-vits/) is a technique for improving the accuracy of block sparsified models by learning a block-sparse mask during a training phase.

```python
from torchao.sparsity import SupermaskLinear, block_sparse_weight
sparsify_(model, lambda x: SupermaskLinear.from_linear(x, block_size=64, sparsity_level=0.9)
# training here

# collapse supermask into a normal linear layer (with many weights set to 0) and then convert to block sparse format for inference speedup
sparsify_(model, lambda x: SupermaskLinear.to_linear(x, sparsity_level=0.9)
sparsify_(model, block_sparse_weight(blocksize=64))
```

#### Dynamic quantization W4A4 CUTLASS-based kernel (https://github.com/pytorch/ao/pull/1515) 

This kernel which adds support for 4 bit dynamic activation + 4 bit weight quantization can be used as follows:

```python
from torchao.quantization import int4_dynamic_activation_int4_weight
quantize_(model, int4_dynamic_activation_int4_weight)
```


## New Contributors
* @jaewoosong made their first contribution in https://github.com/pytorch/ao/pull/1560
* @haodongucsb made their first contribution in https://github.com/pytorch/ao/pull/1630
* @nikhil-arm made their first contribution in https://github.com/pytorch/ao/pull/1447
* @ngc92 made their first contribution in https://github.com/pytorch/ao/pull/1650
* @balancap made their first contribution in https://github.com/pytorch/ao/pull/1667

**Full Changelog**: https://github.com/pytorch/ao/compare/v0.8.0...v0.9.0-rc1