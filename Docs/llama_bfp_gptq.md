# GPTQ 기반 Activation 양자화 오차 Weight 보정

## 1. 목적함수

$$\min_{\Delta W} \| WX - (W + \Delta W) X_Q \|_F^2$$

- $X$: 원본 FP Activation (target 생성용)
- $X_Q$: 양자화된 Activation (BFP)
- $W$: 원본 FP Weight
- $\Delta W$: 보정 행렬 (Weight는 FP 유지)

전개하면:

$$= \text{tr}(\Delta W \cdot X_Q X_Q^T \cdot \Delta W^T) - 2\,\text{tr}(W(X - X_Q) X_Q^T \Delta W^T) + \text{const}$$

---

## 2. Closed-form Solution

$$\Delta W = W(X - X_Q) X_Q^T (X_Q X_Q^T + \lambda I)^{-1}$$

$$W' = W + \Delta W$$

- $\lambda$: regularization (rank-deficiency 방지)

---

## 3. Hessian 변경

GPTQ 대비 유일한 변경점: **Hessian을 $X_Q$ 기준으로 계산**

| | GPTQ | 본 방법 |
|---|---|---|
| Hessian | $H = 2XX^T$ | $H = 2X_Q X_Q^T$ |
| 최적화 변수 | $W_Q$ | $\Delta W$ |
| Target | $WX$ | $WX$ (FP 고정) |

---

## 4. BFP를 고려한 Block-wise 적용

BFP는 block 단위 shared exponent를 사용하므로 **보정도 block 내에서만** 수행.

### 4.1 Block 내 처리 순서

1. Activation magnitude 기준으로 channel 내림차순 정렬
2. 큰 값(salient channel)부터 $\Delta W$ 보정
3. 보정 오차를 나머지 channel에 GPTQ 방식으로 전파
4. $W'_{block}$을 BFP로 quantize → $W_Q$

### 4.2 이유

BFP shared exponent = block 내 최대값 기준이므로, 큰 값부터 보정해야 오차 전파가 최소화됨.

---

## 5. 알고리즘

```
Input:  W (FP), X (FP activation), block_size, λ
Output: W_Q (BFP quantized)

For each BFP block b along channel dim:
    # 1. Block 추출
    W_b  = W[:, b*bs : (b+1)*bs]          # (out, bs)
    X_b  = X[b*bs : (b+1)*bs, :]          # (bs, N)
    
    # 2. Activation quantize
    X_Q_b = quant_BFP(X_b)                # shared exp 결정
    
    # 3. Hessian 계산
    H = 2 * X_Q_b @ X_Q_b.T + λ * I      # (bs, bs)
    H_inv = cholesky_inv(H)               # Cholesky로 안정적 역행렬
    
    # 4. Magnitude 기준 정렬
    order = argsort(mean(|X_b|, dim=1), descending=True)
    
    # 5. Column-wise 보정 (GPTQ 스타일, 정렬 순서대로)
    For each j in order:
        err_act = X_b[j] - X_Q_b[j]       # activation 오차
        δ = W_b[:, j] @ err_act            # (out,)
        
        # 나머지 column에 오차 전파
        W_b[:, j+1:] -= outer(δ, H_inv[j, j+1:]) / H_inv[j, j]
    
    # 6. ΔW 계산 및 보정
    ΔW = W_b @ (X_b - X_Q_b) @ X_Q_b.T @ H_inv
    W_b_corrected = W_b + ΔW
    
    # 7. BFP quantize
    W_Q[:, b*bs:(b+1)*bs] = quant_BFP(W_b_corrected)
```

---

## 6. Python 구현

```python
import torch
import torch.nn.functional as F


def quant_bfp(x: torch.Tensor, block_size: int, mantissa_bits: int) -> torch.Tensor:
    """
    BFP quantization: block 단위 shared exponent
    x: (*, block_size) 마지막 dim이 block
    """
    shape = x.shape
    x_flat = x.reshape(-1, block_size)
    
    # shared exponent: block 내 최대 절댓값 기준
    max_val = x_flat.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
    shared_exp = torch.floor(torch.log2(max_val))
    
    scale = 2 ** (shared_exp - (mantissa_bits - 1))
    x_q = torch.round(x_flat / scale) * scale
    
    return x_q.reshape(shape)


def cholesky_inv(H: torch.Tensor, lambda_reg: float = 1e-4) -> torch.Tensor:
    """Cholesky 기반 안정적 역행렬"""
    H_reg = H + lambda_reg * torch.eye(H.shape[0], device=H.device)
    L = torch.linalg.cholesky(H_reg)
    return torch.cholesky_inverse(L)


def correct_and_quantize(
    W: torch.Tensor,        # (out_features, in_features)
    X: torch.Tensor,        # (in_features, N) calibration data
    block_size: int = 32,
    mantissa_bits: int = 5,
    lambda_reg: float = 1e-4,
) -> torch.Tensor:
    """
    Activation 양자화 오차를 Weight로 보정 후 BFP quantize
    
    목적함수: min_{ΔW} ||WX - (W + ΔW)X_Q||_F^2
    """
    out_features, in_features = W.shape
    W_Q = torch.zeros_like(W)
    
    for b in range(0, in_features, block_size):
        end = min(b + block_size, in_features)
        bs = end - b
        
        W_b = W[:, b:end].clone()       # (out, bs)
        X_b = X[b:end, :].clone()       # (bs, N)
        
        # 1. Activation BFP quantize
        X_Q_b = quant_bfp(X_b.T, bs, mantissa_bits).T  # (bs, N)
        
        # 2. Hessian (X_Q 기준)
        H = 2 * X_Q_b @ X_Q_b.T        # (bs, bs)
        H_inv = cholesky_inv(H, lambda_reg)
        
        # 3. Magnitude 기준 정렬 (내림차순)
        magnitude = X_b.abs().mean(dim=1)
        order = torch.argsort(magnitude, descending=True)
        
        # 4. Column-wise GPTQ 스타일 오차 전파
        W_b_reordered = W_b[:, order].clone()
        X_b_reordered = X_b[order, :]
        X_Q_b_reordered = X_Q_b[order, :]
        
        for i in range(bs):
            err_act = X_b_reordered[i] - X_Q_b_reordered[i]   # (N,)
            delta = W_b_reordered[:, i] @ err_act.unsqueeze(0) # (out, N) → scalar per out
            
            if i + 1 < bs:
                # 나머지 column에 오차 전파
                W_b_reordered[:, i+1:] -= (
                    (W_b_reordered[:, i:i+1] * (err_act.mean())) *
                    H_inv[i, i+1:].unsqueeze(0) / H_inv[i, i]
                )
        
        # 5. ΔW 계산
        act_err = X_b - X_Q_b          # (bs, N)
        delta_W = W_b @ act_err @ X_Q_b.T @ H_inv  # (out, bs)
        W_b_corrected = W_b + delta_W
        
        # 6. BFP quantize
        W_Q[:, b:end] = quant_bfp(W_b_corrected.T, bs, mantissa_bits).T
    
    return W_Q


# 사용 예시
if __name__ == "__main__":
    torch.manual_seed(42)
    
    out_features, in_features, N = 128, 256, 512
    block_size = 32
    
    W = torch.randn(out_features, in_features)
    X = torch.randn(in_features, N)
    
    # 원본 출력
    Y_fp = W @ X
    
    # Activation quantize (오차 발생)
    X_Q = quant_bfp(X.T, block_size, mantissa_bits=5).T
    Y_naive = W @ X_Q  # 보정 없이
    
    # 보정 후 quantize
    W_Q = correct_and_quantize(W, X, block_size=block_size)
    Y_corrected = W_Q @ X_Q
    
    # 오차 비교
    err_naive = (Y_fp - Y_naive).norm() / Y_fp.norm()
    err_corrected = (Y_fp - Y_corrected).norm() / Y_fp.norm()
    
    print(f"보정 전 상대 오차: {err_naive:.4f}")
    print(f"보정 후 상대 오차: {err_corrected:.4f}")
```

---

## 7. 요약

```
Calibration data X 준비
        ↓
X_Q = BFP_quant(X)
        ↓
For each BFP block:
    H = 2 * X_Q @ X_Q.T  ← Hessian (X_Q 기준)
    magnitude 기준 정렬
    column-wise 오차 전파 (GPTQ)
    ΔW = W(X - X_Q) X_Q.T H^{-1}
    W' = W + ΔW
    W_Q = BFP_quant(W')
        ↓
추론: W_Q @ X_Q ≈ W @ X
```