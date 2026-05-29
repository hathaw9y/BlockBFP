# LLaMA Rotation Notes

This note describes the LLaMA rotation flow in `fake_quant/rotation_utils.py`.
It separates offline weight rotations from online activation rotations.

## Rotation Types

QuaRot uses two kinds of rotations:

1. Offline rotations

These are applied once to model weights before evaluation. They rewrite Linear
weights so the model computes in a rotated hidden basis.

2. Online rotations

These are applied during `forward()` to intermediate activations. They are used
where a rotation cannot be fully absorbed into static weights, or where QuaRot
adds an extra Hadamard before quantization.

The two must match. If an online Hadamard is block-wise, the corresponding
weight-side absorbed Hadamard must use the same block size.

## Main Hidden-State Rotation `Q`

`Q` is the main orthogonal hidden-state rotation.

For full Hadamard mode:

```text
Q = D @ H_hidden
```

where `D` is a random sign diagonal matrix and `H_hidden` is a normalized
Hadamard matrix.

For block mode:

```text
Q = block_diag(D_0 @ H_B, D_1 @ H_B, ...)
```

where `B` is the block size. In the current BFP experiments, the default block
size is 32 when `--a_quant_method bfp` and `--rotation_block_size -1`.

## Offline Rotation Map

The hidden state is conceptually changed from `x` to:

```text
x_rot = x @ Q
```

To preserve the model function, adjacent weights are rewritten.

### Embedding

The embedding output is rotated:

```python
embed.weight = embed.weight @ Q
```

Effect:

```text
hidden_states enter the first layer already in rotated basis
```

### Attention Input Projections

For each layer:

```python
q_proj.weight = q_proj.weight @ Q
k_proj.weight = k_proj.weight @ Q
v_proj.weight = v_proj.weight @ Q
```

PyTorch Linear computes `x @ W.T`. If input is `x @ Q`, then using
`W @ Q` preserves the original projection:

```text
(x @ Q) @ (W @ Q).T = x @ Q @ Q.T @ W.T = x @ W.T
```

### Attention Output Projection

For `o_proj`:

```python
o_proj.weight = Q.T @ o_proj.weight
```

This maps the attention output back into the rotated residual basis.

If `o_proj` has bias:

```python
o_proj.bias = Q.T @ o_proj.bias
```

### MLP Input Projections

For LLaMA:

```python
up_proj.weight   = up_proj.weight @ Q
gate_proj.weight = gate_proj.weight @ Q
```

This preserves MLP input projections under the rotated residual stream.

### MLP Output Projection

For `down_proj`:

```python
down_proj.weight = Q.T @ down_proj.weight
```

If `down_proj` has bias:

```python
down_proj.bias = Q.T @ down_proj.bias
```

This maps MLP output back into the rotated residual basis.

### LM Head

The final hidden state is in rotated basis, so:

```python
lm_head.weight = lm_head.weight @ Q
```

Then:

```text
(x @ Q) @ (lm_head.weight @ Q).T = x @ lm_head.weight.T
```

## Online Hadamard Rotations

The main `Q` above is offline. QuaRot also adds Hadamards online before certain
matmuls to reduce activation outliers.

### `down_proj` Input Online Hadamard

Location:

```text
MLP intermediate activation -> online Hadamard -> down_proj
```

Implementation:

```python
qlayers["...down_proj"].online_full_had = True
```

Forward behavior:

```python
x = H_online(x)
x = down_proj(x)
```

Counterpart in weights:

```python
down_proj.weight = down_proj.weight @ H_online
```

In code this is applied after `Q.T @ down_proj.weight`.

If online Hadamard uses block size `B`, the weight-side Hadamard must also use
the same `B`.

### `o_proj` Input Online Hadamard

Location:

```text
attention output -> online Hadamard -> o_proj
```

This can be disabled with:

```text
--no-online_o_proj_had
```

When enabled, forward behavior is:

```python
x = H_online(x)
x = o_proj(x)
```

Counterpart in weights:

```python
o_proj.weight = o_proj.weight @ H_online
```

For the original full/head QuaRot path, `v_proj` is also adjusted by a head-wise
Hadamard:

```python
v_proj output dimension gets H_head
o_proj input dimension gets matching H
```

For the block-BFP path, the current experimental code applies the block
counterpart on `o_proj`. Keep the online transform and weight-side transform
identical.

### Q/K Online Hadamard

When K-cache quantization is enabled, Q and K are rotated after RoPE:

```text
q, k = apply_rotary_pos_emb(q, k)
q = H_online(q)
k = H_online(k)
```

This is not a residual-stream `Q` rotation. It is an attention-space Hadamard
used before K quantization and QK attention.

In the current fake-quant path it is implemented by `QKRotationWrapper`.

## Block Size Rules

`rotation_block_size` controls both offline `Q` and online Hadamards.

```text
--rotation_block_size -1
    If BFP activation is enabled, use BFP block size.
    With default BFP groupsize, this is 32.

--rotation_block_size 0
    Use full hidden-size rotation for offline Q.
    Online paths fall back to the legacy full/head behavior.

--rotation_block_size N
    Use block size N for offline Q and online Hadamards.
```

Block size must divide the dimension being transformed and must be a power of 2.

## Required Invariance Checks

For any online Hadamard `H`, this identity should hold:

```text
x @ W.T == (x @ H) @ (W @ H).T
```

Because normalized Hadamard is orthogonal:

```text
H @ H.T = I
```

So if you add:

```python
x = H(x)
```

before a Linear, you must also rewrite:

```python
linear.weight = linear.weight @ H
```

If `H` is block diagonal, both sides must use the same block diagonal matrix.
Mixing `H_block` online with `H_full` in the weight is not invariant and can
make perplexity explode.

## Recommended Order

For LLaMA:

```text
load model
fuse RMSNorm scales into Linear weights
replace RMSNorm with RMSN
build Q
apply offline Q rotations to weights
apply offline counterparts for online Hadamards
wrap Linear layers for activation quantization
enable online Hadamard flags
evaluate
```

The fuse step should happen before rotation. The activation quant wrappers
should be added after weight rotations so they see the final rotated modules.

## Minimal Mental Model

There are two separate contracts:

1. Residual-basis contract:

```text
all residual stream tensors live in x @ Q space
all weights touching residual stream are rewritten with Q or Q.T
```

2. Online-Hadamard contract:

```text
if an activation is transformed by H immediately before a Linear,
that Linear's input dimension must be rewritten by the same H
```

Most rotation bugs are violations of one of these two contracts.
