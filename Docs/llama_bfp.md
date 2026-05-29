# LLaMA BFP Quantization Notes

This note describes the Block Floating Point (BFP) convention used in the
current LLaMA fake-quant experiments.

## Basic Definition

BFP groups multiple scalar values into a block. Values inside the same block
share one exponent/scale, while each value keeps its own signed integer mantissa.

For a block `x_block`:

```text
scale = shared power-of-two scale for the block
q_i   = round(x_i / scale)
xhat_i = q_i * scale
```

The current implementation is fake quantization: it quantizes and immediately
dequantizes back to floating point. It does not pack or execute real low-bit GEMM.

## Default Block Size

The default BFP block size is:

```text
32 values per block
```

In code this is:

```python
BFP_DEFAULT_BLOCK_SIZE = 32
```

If a BFP quantizer receives `groupsize=-1`, it should use block size 32.

This applies to activation BFP by default:

```text
--a_quant_method bfp --a_groupsize -1  =>  block size 32
```

The same convention should be used for K/V BFP unless a different groupsize is
explicitly provided:

```text
--k_quant_method bfp --k_groupsize -1  =>  block size 32
--v_quant_method bfp --v_groupsize -1  =>  block size 32
```

## What Does "4-bit BFP" Mean Here?

In the current convention, `bits=4` means **sign is included in the 4 bits**.

So 4-bit signed BFP uses:

```text
minq = -8
maxq =  7
```

This is not "4 mantissa bits plus sign". It is total signed 4-bit storage for
each value, plus one shared block scale.

For `bits = b`:

```text
minq = -2^(b - 1)
maxq =  2^(b - 1) - 1
```

## Scale Selection

For each block:

```python
xmax = max(abs(x_block)) * clip_ratio
scale = 2 ** ceil(log2(xmax / maxq))
```

If `xmax == 0`, use:

```python
scale = 1
```

The scale is always a power of two. That is what makes this BFP-like rather
than ordinary affine INT quantization.

Then:

```python
q = clamp(round(x / scale), minq, maxq)
xhat = q * scale
```

## Block Axis

Blocks are formed along the last dimension of the tensor.

For example, if activation shape is:

```text
[batch, seq_len, hidden_size]
```

BFP block size 32 means:

```text
hidden_size is split into consecutive chunks of 32
each chunk gets one shared scale
```

For attention tensors:

```text
Q, K, V:          [batch, heads, seq_len, head_dim]
attention_probs: [batch, heads, q_len, kv_len]
```

BFP blocks are still formed along the last dimension:

```text
Q/K/V blocks are along head_dim
attention probability blocks are along kv_len
```

If the last dimension is not divisible by 32, pad the last dimension to the next
multiple of 32 for scale computation and quantization, then slice back to the
original length.

## Activation BFP

Linear input activation BFP is applied before each wrapped Linear:

```text
activation -> BFP fake-quant -> Linear
```

For LLaMA fake quant, `--a_bits 4 --a_quant_method bfp` means:

```text
4-bit signed BFP
block size 32 unless --a_groupsize is explicitly set
```

`lm_head` input quantization is skipped in the current fake-quant flow.

## V BFP

V-cache/value BFP is implemented through the output quantizer of `v_proj`:

```text
hidden -> v_proj -> V output -> BFP fake-quant
```

Command convention:

```text
--v_bits 4 --v_quant_method bfp
```

If `--v_groupsize -1`, use block size 32.

## K BFP

K BFP is applied after RoPE and Q/K online Hadamard rotation:

```text
q, k = apply_rotary_pos_emb(q, k)
q = H(q)
k = H(k)
k = BFP fake-quant(k)
```

Command convention:

```text
--k_bits 4 --k_quant_method bfp
```

If `--k_groupsize -1`, use block size 32.

## Attention Operation BFP

If experimenting with BFP inside attention matmuls, use the same block
convention:

```text
QK matmul inputs:
    Q -> BFP
    K -> BFP

AV matmul inputs:
    attention_probs -> BFP
    V -> BFP
```

This requires a manual attention path because fused SDPA does not expose the
intermediate attention probabilities.

## Relation To Rotation Block Size

For BFP experiments, rotation block size should usually match BFP block size.

Default:

```text
BFP block size = 32
rotation block size = 32
```

This avoids mixing statistics across a different region than the BFP block.

If using block Hadamard rotation:

```text
Q = block_diag(H_32, H_32, ...)
```

then online Hadamards and the weight-side absorbed Hadamards should also use
block size 32.

## Minimal BFP Function

```python
def bfp_fake_quant(x, bits=4, block_size=32, clip_ratio=1.0):
    minq = -(2 ** (bits - 1))
    maxq = 2 ** (bits - 1) - 1

    orig_shape = x.shape
    pad = (block_size - x.shape[-1] % block_size) % block_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))

    x = x.reshape(-1, x.shape[-1] // block_size, block_size)
    xmax = torch.amax(torch.abs(x), dim=-1, keepdim=True) * clip_ratio
    scale = 2 ** torch.ceil(torch.log2(xmax / maxq))
    scale[xmax == 0] = 1

    q = torch.clamp(torch.round(x / scale), minq, maxq)
    xhat = (q * scale).reshape(*orig_shape[:-1], -1)

    if pad:
        xhat = xhat[..., :orig_shape[-1]]
    return xhat.reshape(orig_shape)
```

## Practical Notes

This implementation is slow compared with real low-bit inference because it is
reference fake quantization. It performs reductions and elementwise operations
at runtime:

```text
amax, log2, ceil, pow, round, clamp, multiply
```

For fast inference, implement the BFP quantization and matmul in CUDA/Triton or
another fused low-level kernel.
