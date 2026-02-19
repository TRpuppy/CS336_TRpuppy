import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
from jaxtyping import Float, Int
from torch import Tensor
from einops import einsum, rearrange, reduce
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        
        # 计算标准差: σ = sqrt(2/(d_in + d_out))
        std = (2.0 / (in_features + out_features)) ** 0.5
        # 截断范围: [-3σ, 3σ]
        a, b = -3 * std, 3 * std
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=a, b=b)

    def forward(self, x: Float[Tensor, " ... in_features"]) -> Float[Tensor, " ... out_features"]:
        Y = einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")
        return Y

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.Embedding_matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.Embedding_matrix, mean=0, std=1, a=-3, b=3)

    def forward(self, x: Int[Tensor, " ..."]) -> Float[Tensor, " ... embedding_dim"]:
        return self.Embedding_matrix[x.int()]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
    def forward(self, x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        RMS = (reduce(x ** 2, " ... d_model -> ... 1", "mean") + self.eps) ** 0.5
        result = einsum(x / RMS, self.weight, " ... d_model, d_model -> ... d_model")
        return result

class SiLU(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()

    def forward(self, x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.silu = SiLU(device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))

class RoPE(nn.Module):
    def __init__(self, Theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.Theta = Theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        #theta_i,k = i/Theta^(2k/d) for k in range(d/2) and i in range(max_seq_len)
        thetas = torch.as_tensor([[i/self.Theta**(2*k/self.d_k)
            for k in range(self.d_k//2)] for i in range(self.max_seq_len)], device=device, dtype=dtype)
        self.register_buffer("cos_cached", torch.cos(thetas), persistent=False) #cos_cached: (max_seq_len, d_k//2)
        self.register_buffer("sin_cached", torch.sin(thetas), persistent=False) #sin_cached: (max_seq_len, d_k//2)

    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: torch.Tensor | None = None) -> Float[Tensor, "... seq_len d_k"]:
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=self.device, dtype=torch.long)
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k//2)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k//2)
        
        # divide x into d_k//2 pairs
        # x: (..., seq_len, d_k)
        # rearrange to: (..., seq_len, d_k//2, 2)
        x_pair = rearrange(x, "... seq_len (d_k_half pair) -> ... seq_len d_k_half pair", pair=2, d_k_half=self.d_k//2)
        # for each pair (x_0, x_1), apply rotation to get (x_0*cos - x_1*sin, x_0*sin + x_1*cos)
        x_rotated = torch.stack([
            x_pair[..., 0] * cos - x_pair[..., 1] * sin,
            x_pair[..., 0] * sin + x_pair[..., 1] * cos
        ], dim=-1)  # (..., seq_len, d_k//2, 2)
        
        # rearrange back to original shape (..., seq_len, d_k)
        x_out = rearrange(x_rotated, "... seq_len d_k_half pair -> ... seq_len (d_k_half pair)", pair=2, d_k_half=self.d_k//2)
        return x_out
        

def Softmax(x: Float[Tensor, " ..."] , dim: int) -> Float[Tensor, " ..."]:
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

class scaled_dot_product_attention(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self,
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Float[Tensor, "queries keys"] | None = None) -> Float[Tensor, " ... queries d_v"]:
        if mask is None:
            # Create mask without batch dimensions: (queries, keys)
            mask = torch.ones(Q.shape[-2], K.shape[-2], device=self.device, dtype=self.dtype)
        d_k = Q.shape[-1]
        QK_norm = einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys") / (d_k ** 0.5)
        QK_norm = QK_norm.masked_fill(mask == 0, -torch.inf)
        QK_norm = Softmax(QK_norm, dim=-1)
        return einsum(QK_norm, V, " ... queries keys, ... keys d_v -> ... queries d_v")

class multihead_self_attention(nn.Module):
    def __init__(self, d_in: int, d_out: int, num_heads: int, d_k: int, d_v: int, RoPE: nn.Module | None = None , token_positions: torch.Tensor | None = None , device=None, dtype=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.d_k = d_k
        self.d_v = d_v
        self.sdp_attention = scaled_dot_product_attention() #d_k = d_model//num_heads
        self.RoPE = RoPE
        self.token_positions = token_positions
        self.W_q = nn.Parameter(torch.empty(num_heads * d_k, d_in, device=device, dtype=dtype))
        self.W_k = nn.Parameter(torch.empty(num_heads * d_k, d_in, device=device, dtype=dtype))
        self.W_v = nn.Parameter(torch.empty(num_heads * d_v, d_in, device=device, dtype=dtype))
        self.W_o = nn.Parameter(torch.empty(d_out, num_heads * d_v, device=device, dtype=dtype))


        std_W_q = (2.0 / (self.W_q.shape[0] * self.W_q.shape[1])) ** 0.5
        nn.init.trunc_normal_(self.W_q, mean=0.0, std=std_W_q, a=-3 * std_W_q, b=3 * std_W_q)
        std_W_k = (2.0 / (self.W_k.shape[0] * self.W_k.shape[1])) ** 0.5
        nn.init.trunc_normal_(self.W_k, mean=0.0, std=std_W_k, a=-3 * std_W_k, b=3 * std_W_k)
        std_W_v = (2.0 / (self.W_v.shape[0] * self.W_v.shape[1])) ** 0.5
        nn.init.trunc_normal_(self.W_v, mean=0.0, std=std_W_v, a=-3 * std_W_v, b=3 * std_W_v)
        std_W_o = (2.0 / (self.W_o.shape[0] * self.W_o.shape[1])) ** 0.5
        nn.init.trunc_normal_(self.W_o, mean=0.0, std=std_W_o, a=-3 * std_W_o, b=3 * std_W_o)
    
    def forward(self, x: Float[Tensor, " ... sequence_length d_in"]) -> Float[Tensor, " ... sequence_length d_out"]:
        Q = einsum(x, self.W_q, " ... sequence_length d_in, d d_in -> ... sequence_length d")#d = d_k * h
        Q = rearrange(Q, " ... sequence_length (h d_k) -> ... h sequence_length d_k", h=self.num_heads , d_k = self.d_k)
        if self.RoPE is not None:
            Q = self.RoPE(Q, self.token_positions)
        K = einsum(x, self.W_k, " ... sequence_length d_in, d d_in -> ... sequence_length d")#d = d_k * h
        K = rearrange(K, " ... sequence_length (h d_k) -> ... h sequence_length d_k", h=self.num_heads , d_k = self.d_k)
        if self.RoPE is not None:
            K = self.RoPE(K, self.token_positions)
        V = einsum(x, self.W_v, " ... sequence_length d_in, d d_in -> ... sequence_length d")#d = d_v * h
        V = rearrange(V, " ... sequence_length (h d_v) -> ... h sequence_length d_v", h=self.num_heads , d_v = self.d_v)
        
        mask = torch.tril(torch.ones(Q.shape[-2], Q.shape[-2], device=self.device, dtype=self.dtype))
        Result = self.sdp_attention(Q, K, V, mask)
        Result = rearrange(Result, " ... h sequence_length d_v -> ... sequence_length (h d_v)", h=self.num_heads , d_v = self.d_v)
        Result = einsum(Result, self.W_o, " ... sequence_length d, d_model d -> ... sequence_length d_model")#d = d_v * h
        return Result

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.msa_RoPE = RoPE(theta, d_model//num_heads, max_seq_len, device=device, dtype=dtype)
        self.msa = multihead_self_attention(d_in=d_model, d_out=d_model, num_heads=num_heads, d_k=d_model // num_heads, d_v=d_model // num_heads, RoPE=self.msa_RoPE, device=device, dtype=dtype)
        self.rms_norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.swiglu = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.rms_norm2 = RMSNorm(d_model, device=device, dtype=dtype)
    
    def _msa_forward(self, x: Float[Tensor, " ... sequence_length d_model"]) -> Float[Tensor, " ... sequence_length d_model"]:
        """MSA 部分的 forward，用于梯度检查点"""
        y = self.rms_norm1(x)
        y = self.msa(y)
        return y
    
    def _swiglu_forward(self, x: Float[Tensor, " ... sequence_length d_model"]) -> Float[Tensor, " ... sequence_length d_model"]:
        """SwiGLU 部分的 forward，用于梯度检查点"""
        y = self.rms_norm2(x)
        y = self.swiglu(y)
        return y
    
    def forward(self, x: Float[Tensor, " ... sequence_length d_model"]) -> Float[Tensor, " ... sequence_length d_model"]:
        # 检查点1：MSA 前（保存 rms_norm1 的输出作为检查点）
        # 在反向传播时，如果需要中间激活值，会从检查点重新计算 rms_norm1 + msa
        #y = checkpoint(self._msa_forward, x, use_reentrant=False)
        y = self._msa_forward(x)
        x = x + y
        
        # 检查点2：SwiGLU 前（保存 rms_norm2 的输出作为检查点）
        # 在反向传播时，如果需要中间激活值，会从检查点重新计算 rms_norm2 + swiglu
        #y = checkpoint(self._swiglu_forward, x, use_reentrant=False)
        y = self._swiglu_forward(x)
        return x + y

class CrossEntropyLoss(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x: Float[Tensor, " ... batch_size vocab_size"] , targets: Int[Tensor, " ... batch_size"]) -> Float[Tensor, "..."]:
        max_x = x.max(dim=-1, keepdim=True).values
        x = x - max_x
        sum_exp_x = torch.exp(x).sum(dim=-1, keepdim=False)
        log_sum_exp_x = torch.log(sum_exp_x)
        # 使用 gather 从每个样本中选择目标类别的 logit
        x_targets = x.gather(-1, targets.int().unsqueeze(-1)).squeeze(-1)
        losses = log_sum_exp_x - x_targets
        return losses.mean()

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.rms_norm_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, rope_theta, max_seq_len, device=device, dtype=dtype) for _ in range(num_layers)])
        self.Linear_final = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.cross_entropy = CrossEntropyLoss(device=device, dtype=dtype)
    def forward(self, x: Int[Tensor, " ... sequence_length"], targets: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, "..."]:
        x = self.token_embeddings(x)
        # TransformerBlock 内部已经设置了更细粒度的检查点（MSA 前和 SwiGLU 前）
        # 因此这里直接调用即可，不需要额外的检查点包装
        for layer in self.layers:
            # x = layer(x)
            x = checkpoint(layer, x, use_reentrant=False)
        x = self.rms_norm_final(x)
        x = self.Linear_final(x)
        loss = self.cross_entropy(x, targets)
        return loss
    
    def generate(self, x: Int[Tensor, " ... sequence_length"], temperature: float = 0.7, top_p: float = 0.9) -> Int[Tensor, " ..."]:
        if x.shape[-1] > self.max_seq_len:
            x = x[..., -self.max_seq_len:]
        
        x = self.token_embeddings(x)
        # 通过所有 TransformerBlock 层
        for layer in self.layers:
            x = layer(x)
        # 通过最终的 RMSNorm 和 Linear 层
        x = self.rms_norm_final(x)
        x = self.Linear_final(x)
        # 取最后一个位置的 logits，形状为 (..., vocab_size)
        logits = x[..., -1, :]
        

        scaled_logits = logits / temperature
        probs = Softmax(scaled_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        # 计算累积概率
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        exceeds_threshold = cumsum_probs >= top_p
        # 如果没有超过阈值的，保留所有
        num_to_keep = exceeds_threshold.long().argmax(dim=-1) + 1
        no_exceed = ~exceeds_threshold.any(dim=-1)
        num_to_keep = torch.where(no_exceed, torch.tensor(sorted_probs.shape[-1], device=sorted_probs.device), num_to_keep)
        indices = torch.arange(sorted_probs.shape[-1], device=sorted_probs.device)
        if sorted_probs.dim() > 1:
            num_to_keep = num_to_keep.unsqueeze(-1)
            indices = indices.unsqueeze(0).expand_as(sorted_probs)
            mask = indices < num_to_keep
        else:
            mask = indices < num_to_keep
        filtered_probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        next_token_idx = torch.multinomial(filtered_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, next_token_idx).squeeze(-1)
        
        return next_token


