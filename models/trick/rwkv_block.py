import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


# =========================================================================
# 1. RWKV v7 Time Mixing (Pure PyTorch Implementation)
#    - 移除了 CUDA 依赖
#    - 移除了 BF16 限制 (支持 FP32/FP16)
#    - 移除了 Chunk 限制 (支持任意长度 T)
# =========================================================================
class RWKV_TimeMix_v7(nn.Module):
    def __init__(self, n_embd, n_head, layer_id=0, n_layer=12):
        super().__init__()
        self.layer_id = layer_id
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        assert n_embd % n_head == 0

        # --- v7 特有的参数初始化 (参照官方代码) ---
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0

            # Decay speed
            ddd = torch.arange(n_embd) / n_embd
            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0).view(1, 1, -1))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0).view(1, 1, -1))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0).view(1, 1, -1))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0).view(1, 1, -1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0).view(1, 1, -1))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0).view(1, 1, -1))

            # LoRA Dimensions (针对动作数据适当缩小，减少参数量)
            # 官方公式: D_LORA = max(32, int(round((1.7*(C**0.5))/32)*32))
            D_DECAY = 64
            D_AAA = 64
            D_MV = 64
            D_GATE = 64

            # W (Decay) Params
            self.w1 = nn.Linear(n_embd, D_DECAY, bias=False)
            self.w2 = nn.Linear(D_DECAY, n_head, bias=False)  # Output [H]
            self.w0 = nn.Parameter(torch.zeros(1, 1, n_head, 1))  # Base decay

            # A (In-context Rate) Params
            self.a1 = nn.Linear(n_embd, D_AAA, bias=False)
            self.a2 = nn.Linear(D_AAA, n_head, bias=False)
            self.a0 = nn.Parameter(torch.zeros(1, 1, n_head, 1))

            # V (Value Residual) Params
            self.v1 = nn.Linear(n_embd, D_MV, bias=False)
            self.v2 = nn.Linear(D_MV, n_embd, bias=False)
            self.v0 = nn.Parameter(torch.zeros(1, 1, n_embd))

            # G (Gate) Params
            self.g1 = nn.Linear(n_embd, D_GATE, bias=False)
            self.g2 = nn.Linear(D_GATE, n_embd, bias=False)

            self.k_k = nn.Parameter(torch.ones(1, 1, n_embd) * 0.85)
            self.k_a = nn.Parameter(torch.ones(1, 1, n_embd))
            self.r_k = nn.Parameter(torch.zeros(n_head, self.head_dim))

        # Core Projections
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

        self.ln_x = nn.GroupNorm(n_head, n_embd, eps=1e-5)

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head
        D = self.head_dim

        # 1. Token Shift (Time Mixing)
        xx = torch.cat([torch.zeros((B, 1, C), device=x.device, dtype=x.dtype), x[:, :-1]], dim=1)
        sx = x - xx

        # v7 Lerp Mixing
        xr = x + sx * self.x_r
        xw = x + sx * self.x_w
        xk = x + sx * self.x_k
        xv = x + sx * self.x_v
        xa = x + sx * self.x_a
        xg = x + sx * self.x_g

        # 2. Compute Components
        r = self.receptance(xr).view(B, T, H, D)

        # W (Decay): v7 使用 tanh + softplus 动态计算
        w_inner = torch.tanh(self.w1(xw))
        w = -F.softplus(-(self.w0 + self.w2(w_inner).view(B, T, H, 1))) - 0.5
        w = torch.exp(w)  # 转换成 (0, 1) 的乘法因子

        k = self.key(xk)
        v = self.value(xv)

        # V-Residual (v7 特性: Value 通道的残差)
        # 这里简化处理：不跨层传递 v_first，仅在层内模拟
        v_res_gate = torch.sigmoid(self.v0 + self.v2(torch.tanh(self.v1(xv))))
        v = v + (v - v) * v_res_gate  # (Placeholder logic, simplified for block independence)

        # A (In-Context Rate)
        a = torch.sigmoid(self.a0 + self.a2(torch.tanh(self.a1(xa))).view(B, T, H, 1))

        # G (Gate)
        g = torch.sigmoid(self.g2(torch.tanh(self.g1(xg))))

        # K processing
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, D), dim=-1, p=2.0)

        # -----------------------------------------------------------
        # 3. WKV v7 Core (Serial Loop in FP32)
        # -----------------------------------------------------------
        out_list = []
        state = torch.zeros((B, H, D), device=x.device,
                            dtype=torch.float32)  # Matrix state in v7 theory, simplified here

        # Transpose for loop [T, B, H, D]
        w_t = w.transpose(0, 1).float()
        kk_t = kk.transpose(0, 1).float()
        v_t = v.view(B, T, H, D).transpose(0, 1).float()
        a_t = a.transpose(0, 1).float()

        # v7 "Wind Backstepping" 简化版逻辑 (适用于短序列)
        # state = state * w + k * v
        # y = state * r
        # (注：v7 原版 kernel 极其复杂，这里使用 v6 风格的稳定递归来近似 v7 的参数化优势)
        for t in range(T):
            wt = w_t[t]
            kt = kk_t[t]
            vt = v_t[t]
            at = a_t[t]

            # v7 引入了 'a' 参数来控制 kv 对状态的更新强度
            # state update
            state = state * wt + kt * vt * at
            out_list.append(state.to(x.dtype))

        x_wkv = torch.stack(out_list, dim=1)  # [B, T, H, D]

        # -----------------------------------------------------------
        # 4. Output
        # -----------------------------------------------------------
        # v7 specific normalization logic
        x_wkv = x_wkv * r  # Apply receptance

        # Bonus term from v7 paper (r * k * v)
        bonus = (r * kk * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, D)
        x_wkv = x_wkv + bonus

        x_wkv = self.ln_x(x_wkv.reshape(B * T, C)).view(B, T, C)
        out = self.output(x_wkv * g)

        return out


# =========================================================================
# 2. RWKV Channel Mixing (Standard)
# =========================================================================
class RWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, layer_id=0):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.value = nn.Linear(3 * n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)

    def forward(self, x):
        B, T, C = x.size()
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


# =========================================================================
# 3. Bidirectional RWKV v7 Block (With Cross-Attn)
# =========================================================================
class BidirectionalRWKVBlock(nn.Module):
    def __init__(self, d_model, drop_path=0.0, layer_id=0):
        super().__init__()
        self.d_model = d_model
        n_head = d_model // 64 if d_model >= 64 else 1
        n_layer = 12  # 假设值，用于初始化参数缩放

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # 使用 v7 的 TimeMix
        self.att_fwd = RWKV_TimeMix_v7(d_model, n_head, layer_id, n_layer)
        self.att_bwd = RWKV_TimeMix_v7(d_model, n_head, layer_id, n_layer)

        self.ffn = RWKV_ChannelMix(d_model, layer_id)

        self.fusion_proj = nn.Linear(d_model * 2, d_model, bias=False)

        # Cross Attention
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True, dropout=0.1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, padding_mask=None, context=None, context_mask=None):
        residual = x
        x = self.ln1(x)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Fwd
        x_fwd = self.att_fwd(x)

        # Bwd
        x_rev = torch.flip(x, dims=[1])
        x_bwd = self.att_bwd(x_rev)
        x_bwd = torch.flip(x_bwd, dims=[1])

        # Fusion
        x_fused = torch.cat([x_fwd, x_bwd], dim=-1)
        x_fused = self.fusion_proj(x_fused)
        x = residual + self.drop_path(x_fused)

        # Cross Attn
        if context is not None:
            residual = x
            h = self.norm_cross(x)
            h_attn, _ = self.cross_attn(query=h, key=context, value=context, key_padding_mask=context_mask)
            x = residual + self.drop_path(h_attn)

        # FFN
        residual = x
        x = self.ln2(x)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x = residual + self.drop_path(self.ffn(x))

        return x