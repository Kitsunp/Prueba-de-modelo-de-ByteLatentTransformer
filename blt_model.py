# blt_model_sin_cache_con_debug.py

import torch

# Habilitar Flash SDP explícitamente
torch.backends.cuda.enable_flash_sdp(enabled=True)

# Verificar si está habilitado
print(f"Flash SDP habilitado: {torch.backends.cuda.flash_sdp_enabled()}")

import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Dict, List

# =============================================================================
#                               BLOQUES BÁSICOS
# =============================================================================

class RMSNorm(nn.Module):
    """
    Normalización RMS (Root Mean Square) utilizada como alternativa
    a LayerNorm. Escala la norma RMS de cada vector a 1.
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        print("RMSNorm - Input shape:", x.shape)
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        out = x / norm * self.g
        print("RMSNorm - Output shape:", out.shape)
        return out

class RotaryEmbedding(nn.Module):
    """
    Implementación de Rotary Embeddings para inyectar información posicional
    en las consultas y claves de la atención.
    """
    def __init__(self, dim, theta=500000):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, positions):
        """
        Calcula cosenos y senos correspondientes a las frecuencias rotatorias.
        """
        print("RotaryEmbedding - Positions shape:", positions.shape)
        positions = positions.unsqueeze(-1)
        freqs = positions.float() * self.inv_freq.unsqueeze(0)
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        return freqs_cos, freqs_sin

    def rotate_queries_and_keys(self, q, k, positions):
        """
        Aplica la rotación de RoPE a queries (q) y keys (k).
        """
        print("RotaryEmbedding - rotate Q/K shapes:", q.shape, k.shape)
        freqs_cos, freqs_sin = self.forward(positions)

        batch_size, num_heads, seq_length, head_dim = q.shape
        dim_half = head_dim // 2

        freqs_cos = freqs_cos[:seq_length, :dim_half]
        freqs_sin = freqs_sin[:seq_length, :dim_half]
        
        freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(0)
        freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(0)

        q1, q2 = q[..., :dim_half], q[..., dim_half:]
        k1, k2 = k[..., :dim_half], k[..., dim_half:]

        q_rotate = torch.cat([
            q1 * freqs_cos - q2 * freqs_sin,
            q2 * freqs_cos + q1 * freqs_sin
        ], dim=-1)

        k_rotate = torch.cat([
            k1 * freqs_cos - k2 * freqs_sin,
            k2 * freqs_cos + k1 * freqs_sin
        ], dim=-1)

        print("RotaryEmbedding - Rotated Q/K shapes:", q_rotate.shape, k_rotate.shape)
        return q_rotate, k_rotate

class HeadwiseNorm(nn.Module):
    """
    Normalización específica por cabeza para atención multi-cabeza.
    Normaliza cada cabeza de atención de forma independiente.
    """
    def __init__(self, num_heads, head_dim, eps=1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.beta = nn.Parameter(torch.zeros(num_heads, 1, 1))

    def forward(self, x):
        """
        Input: x de forma [batch_size, num_heads, seq_length, head_dim].
        """
        print("HeadwiseNorm - Input shape:", x.shape)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta
        print("HeadwiseNorm - Output shape:", out.shape)
        return out


# =============================================================================
#                         ATENCIÓN MULTI-CABEZA
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention con múltiples niveles de dropout y Rotary Embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        assert self.head_dim * self.num_heads == self.hidden_size, "hidden_size must be divisible by num_heads"

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Módulos adicionales
        self.rotary = RotaryEmbedding(self.head_dim)  
        self.norm = RMSNorm(config.hidden_size)
        self.head_norm = HeadwiseNorm(num_heads=self.num_heads, head_dim=self.head_dim)

        # Parámetros de dropout
        self.attention_dropout = config.attention_dropout
        self.resid_dropout = nn.Dropout(config.resid_dropout)
        self.proj_dropout = nn.Dropout(config.resid_dropout)
        self.lambda_dropout = nn.Dropout(self.attention_dropout)

        # Inicializar parámetros relacionados con lambda
        self._initialize_lambda_parameters(config)

    def _initialize_lambda_parameters(self, config):
        layer_idx = getattr(config, 'layer_idx', 1)
        base_lambda = 0.8 - 0.6 * math.exp(-0.3 * (layer_idx - 1))
        self.lambda_init = nn.Parameter(torch.full((1, self.num_heads), base_lambda))

        dim_scale = 0.01 / math.sqrt(self.hidden_size)
        self.lambda_q1 = nn.Parameter(torch.randn(1, self.num_heads, self.head_dim) * dim_scale)
        self.lambda_k1 = nn.Parameter(torch.randn(1, self.num_heads, self.head_dim) * dim_scale)
        self.lambda_q2 = nn.Parameter(torch.randn(1, self.num_heads, self.head_dim) * dim_scale)
        self.lambda_k2 = nn.Parameter(torch.randn(1, self.num_heads, self.head_dim) * dim_scale)

    def compute_lambda(self):
        qk1 = torch.sum(self.lambda_dropout(self.lambda_q1) * self.lambda_k1, dim=-1)
        qk2 = torch.sum(self.lambda_dropout(self.lambda_q2) * self.lambda_k2, dim=-1)
        lambda_val = torch.exp(qk1) - torch.exp(qk2) + self.lambda_init
        return torch.clamp(lambda_val, min=0.0, max=1.0)

    def forward(self, x, mask=None, positions=None, is_causal=False):
        print("\n[MultiHeadAttention] - Input X shape:", x.shape)
        batch_size, seq_length, _ = x.size()
        x_norm = self.norm(x)

        print("[MultiHeadAttention] - After RMSNorm X shape:", x_norm.shape)

        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        print("[MultiHeadAttention] - Q/K/V projected shapes:", q.shape, k.shape, v.shape)

        def reshape_to_heads(tensor):
            return tensor.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape_to_heads(q), reshape_to_heads(k), reshape_to_heads(v)
        print("[MultiHeadAttention] - Q/K/V reshaped to heads:", q.shape, k.shape, v.shape)

        # Dividir las cabezas en dos grupos
        q1, q2 = torch.chunk(q, 2, dim=1)
        k1, k2 = torch.chunk(k, 2, dim=1)
        v1, v2 = torch.chunk(v, 2, dim=1)

        # Aplicar RoPE
        q1, k1 = self.rotary.rotate_queries_and_keys(q1, k1, positions)
        q2, k2 = self.rotary.rotate_queries_and_keys(q2, k2, positions)

        # Dropout tras RoPE
        q1, k1 = self.proj_dropout(q1), self.proj_dropout(k1)
        q2, k2 = self.proj_dropout(q2), self.proj_dropout(k2)

        if mask is not None:
            # Convertir mask a boolean con dropout
            mask = self.resid_dropout(mask.float()).bool()

        # Atención
        attn1 = F.scaled_dot_product_attention(
            q1, k1, v1, dropout_p=self.attention_dropout, is_causal=is_causal
        )
        attn2 = F.scaled_dot_product_attention(
            q2, k2, v2, dropout_p=self.attention_dropout, is_causal=is_causal
        )
        print("[MultiHeadAttention] - attn1/attn2 shapes:", attn1.shape, attn2.shape)

        attn1 = self.proj_dropout(attn1)
        attn2 = self.proj_dropout(attn2)

        # Calcular lambda para combinar
        lambda_val = self.compute_lambda()[:, :self.num_heads//2].unsqueeze(-1).unsqueeze(-1)
        out = attn1 - lambda_val * attn2

        # Concatenar cabezas
        out = torch.cat([out, out], dim=1)
        print("[MultiHeadAttention] - Concat attn shape:", out.shape)

        # Normalización por cabeza y dropout posterior
        out = self.head_norm(out)
        out = self.resid_dropout(out)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        out = self.o_proj(out)
        out = self.resid_dropout(out)
        print("[MultiHeadAttention] - Output shape:", out.shape)
        return out

class CrossAttention(nn.Module):
    """
    Atención cruzada (CrossAttention) para mezclar el contexto externo (context)
    con la entrada actual (x).
    """
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.norm = RMSNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout)
        self.proj_dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x, context, patch_mask=None):
        print("\n[CrossAttention] - Input X shape:", x.shape)
        batch_size, seq_length, _ = x.size()
        context_length = context.size(1)

        x_norm = self.norm(x)
        print("[CrossAttention] - After RMSNorm X shape:", x_norm.shape)

        q = self.q_proj(x_norm)
        k = self.k_proj(context)
        v = self.v_proj(context)

        print("[CrossAttention] - Q/K/V shapes:", q.shape, k.shape, v.shape)
        q = self.proj_dropout(q)
        k = self.proj_dropout(k)
        v = self.proj_dropout(v)

        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)

        print("[CrossAttention] - Q/K/V reshaped to heads:", q.shape, k.shape, v.shape)

        attn_mask = None
        if patch_mask is not None:
            attn_mask = (patch_mask == 0)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )
        print("[CrossAttention] - Attn output shape:", out.shape)

        out = self.dropout(out)
        
        out = out.transpose(1, 2).reshape(batch_size, seq_length, self.hidden_size)
        out = self.o_proj(out)
        out = self.dropout(out)
        print("[CrossAttention] - Output shape:", out.shape)
        return out

class FeedForward(nn.Module):
    """
    Capa FeedForward con activación SwiGLU y múltiples dropouts.
    """
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size)
        
        self.norm = RMSNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.resid_dropout)
        self.activation_dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x):
        print("\n[FeedForward] - Input shape:", x.shape)
        x = self.norm(x)
        print("[FeedForward] - After RMSNorm shape:", x.shape)
        swish = F.silu(self.w1(x))
        gate = self.w3(x)
        x = swish * gate
        x = self.activation_dropout(x)
        x = self.w2(x)
        x = self.dropout(x)
        print("[FeedForward] - Output shape:", x.shape)
        return x

class EncoderLayer(nn.Module):
    """
    Capa de encoder que combina self-attention, cross-attention (opcional)
    y feed-forward.
    """
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.cross_attn = CrossAttention(config)
        self.feed_forward = FeedForward(config)
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x, cross_context=None, self_mask=None, cross_mask=None, positions=None):
        print("\n[EncoderLayer] - Input shape:", x.shape)
        h = x + self.self_attn(x, mask=self_mask, positions=positions, is_causal=False)
        print("[EncoderLayer] - After Self-Attn shape:", h.shape)
        h = self.dropout(h)
        
        if cross_context is not None:
            h = h + self.cross_attn(h, cross_context, cross_mask)
            print("[EncoderLayer] - After Cross-Attn shape:", h.shape)
            h = self.dropout(h)
        
        out = h + self.feed_forward(h)
        print("[EncoderLayer] - After FeedForward shape:", out.shape)
        out = self.dropout(out)
        return out

class DecoderLayer(nn.Module):
    """
    Capa de decoder con cross-attention, self-attention con enmascaramiento causal
    y feed-forward.
    """
    def __init__(self, config):
        super().__init__()
        self.cross_attn = CrossAttention(config)
        self.self_attn = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None, positions=None):
        print("\n[DecoderLayer] - Input shape:", x.shape)
        h = x + self.cross_attn(x, encoder_output, cross_mask)
        print("[DecoderLayer] - After Cross-Attn shape:", h.shape)
        h = self.dropout(h)
        
        h = h + self.self_attn(h, self_mask, positions, is_causal=True)
        print("[DecoderLayer] - After Self-Attn shape:", h.shape)
        h = self.dropout(h)
        
        out = h + self.feed_forward(h)
        print("[DecoderLayer] - After FeedForward shape:", out.shape)
        out = self.dropout(out)
        return out


# =============================================================================
#                          EMBEDDINGS A NIVEL DE BYTE
# =============================================================================

class ByteEmbedding(nn.Module):
    """
    Genera embeddings a nivel de byte e incluye n-gram embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.byte_embeddings = nn.Embedding(256, config.hidden_size)
        self.ngram_hash_embeddings = nn.ModuleList([
            nn.Embedding(config.ngram_vocab_size, config.hidden_size)
            for _ in range(6)  # Para n-gramas de tamaño 3 a 8
        ])
        
        self.dropout = nn.Dropout(config.resid_dropout)

    def compute_ngram_hash(self, bytes_sequence, n):
        device = bytes_sequence.device
        batch_size, seq_length = bytes_sequence.shape

        if seq_length < n:
            # Retorna un tensor vacío si la secuencia es más corta que n
            return torch.empty((batch_size, 0), dtype=torch.long, device=device)

        ngrams = bytes_sequence.unfold(dimension=1, size=n, step=1)
        exponents = torch.arange(n, device=device).float()
        weights = (256 ** exponents).unsqueeze(0).unsqueeze(0)

        hash_values = (ngrams.float() * weights).sum(dim=-1).long()
        hash_tensor = hash_values % self.ngram_hash_embeddings[n-3].num_embeddings
        return hash_tensor

    def forward(self, bytes_input):
        print("\n[ByteEmbedding] - Input shape:", bytes_input.shape)
        device = bytes_input.device
        batch_size, seq_length = bytes_input.shape

        embeds = self.byte_embeddings(bytes_input).float()
        embeds = self.dropout(embeds)
        print("[ByteEmbedding] - After byte embedding shape:", embeds.shape)
        
        for n in range(3, 9):
            if seq_length >= n:
                ngram_hashes = self.compute_ngram_hash(bytes_input, n)
                ngram_embeds = self.ngram_hash_embeddings[n-3](ngram_hashes)
                
                expanded_embeds = torch.zeros_like(embeds)
                expanded_embeds[:, :seq_length - n + 1, :] += ngram_embeds / n
                embeds = embeds + expanded_embeds
        print("[ByteEmbedding] - Output shape:", embeds.shape)
        return embeds


# =============================================================================
#                        MODELOS DE ENCODER Y DECODER
# =============================================================================

class LocalEncoder(nn.Module):
    """
    Encoder local que procesa los bytes de forma detallada.
    """
    def __init__(self, config):
        super().__init__()
        self.byte_embeddings = ByteEmbedding(config)
        self.embedding_dropout = nn.Dropout(config.resid_dropout)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, bytes_input, patch_boundaries=None):
        print("\n[LocalEncoder] - Input shape:", bytes_input.shape)
        h = self.byte_embeddings(bytes_input)
        print("[LocalEncoder] - After ByteEmbedding shape:", h.shape)
        h = self.embedding_dropout(h)
        
        positions = torch.arange(bytes_input.size(1), device=bytes_input.device)
        
        for idx, layer in enumerate(self.layers):
            print(f"[LocalEncoder] - Passing through EncoderLayer {idx}")
            h = layer(h, positions=positions)
            print(f"[LocalEncoder] - EncoderLayer {idx} output shape:", h.shape)
            h = self.dropout(h)
        print("[LocalEncoder] - Final output shape:", h.shape)
        return h

class GlobalTransformer(nn.Module):
    """
    Procesa la información a nivel de parches con atención global.
    """
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.global_layers)])

        self.dropout = nn.Dropout(config.resid_dropout)
        self.adaptive_dropout = nn.Dropout(0.08)
        self.gate_dropout = nn.Dropout(0.08)
        self.mem_dropout = nn.Dropout(0.08)
        self.skip_dropout = nn.Dropout(0.08)
        
        self.adaptive_weights = nn.Parameter(torch.ones(config.global_layers))
        self.layer_norms = nn.ModuleList([RMSNorm(config.hidden_size) 
                                          for _ in range(config.global_layers)])
        
        self.skip_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Dropout(0.08),
                nn.Sigmoid()
            ) for _ in range(config.global_layers)
        ])
        
        self.hierarchical_mem = nn.Parameter(
            torch.zeros(config.global_layers, config.hidden_size)
        )
        self.mem_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Dropout(0.08),
            nn.Sigmoid()
        )
        
        self.input_norm = RMSNorm(config.hidden_size)
        self.output_norm = RMSNorm(config.hidden_size)

    def _apply_skip_connection(self, x, residual, layer_idx):
        print("[GlobalTransformer] - _apply_skip_connection")
        if x.dim() != residual.dim():
            residual = residual.view_as(x)
            
        norm_x = self.layer_norms[layer_idx](x)
        norm_residual = self.layer_norms[layer_idx](residual)
        
        gate_input = torch.cat([norm_x, norm_residual], dim=-1)
        gate = self.skip_gates[layer_idx](gate_input)
        gate = self.gate_dropout(gate)
        
        adaptive_weight = self.adaptive_dropout(self.adaptive_weights[layer_idx])
        adaptive_weight = adaptive_weight.view(1, 1, 1)
        
        weighted_residual = adaptive_weight * gate * norm_residual
        weighted_residual = self.skip_dropout(weighted_residual)
        
        out = x + weighted_residual
        print("[GlobalTransformer] - skip_connection output shape:", out.shape)
        return out

    def _apply_hierarchical_memory(self, x, layer_idx, batch_size):
        print("[GlobalTransformer] - _apply_hierarchical_memory")
        mem = self.hierarchical_mem[layer_idx:layer_idx+1]
        mem = mem.unsqueeze(0).expand(batch_size, x.size(1), -1)
        mem = self.mem_dropout(mem)
        
        mem_gate_input = torch.cat([x, mem], dim=-1)
        mem_gate = self.mem_gate(mem_gate_input)
        
        out = x + self.mem_dropout(mem_gate * mem)
        print("[GlobalTransformer] - hierarchical_memory output shape:", out.shape)
        return out

    def forward(self, patch_embeddings, attention_mask=None):
        print("\n[GlobalTransformer] - Input patch_embeddings shape:", patch_embeddings.shape)
        batch_size = patch_embeddings.size(0)
        h = self.dropout(self.input_norm(patch_embeddings))
        print("[GlobalTransformer] - After input_norm shape:", h.shape)
        
        positions = torch.arange(
            patch_embeddings.size(1), 
            device=patch_embeddings.device
        )

        for idx, layer in enumerate(self.layers):
            print(f"[GlobalTransformer] - Passing through EncoderLayer {idx}")
            prev_h = self.dropout(h)
            h = layer(h, self_mask=attention_mask, positions=positions)
            h = self._apply_hierarchical_memory(h, idx, batch_size)
            h = self._apply_skip_connection(h, prev_h, idx)
            h = self.dropout(h)
            print(f"[GlobalTransformer] - EncoderLayer {idx} output shape:", h.shape)
        
        h = self.dropout(self.output_norm(h))
        print("[GlobalTransformer] - Final output shape:", h.shape)
        return h

class LocalDecoder(nn.Module):
    """
    Decoder local que reconvierte las representaciones latentes en logits de bytes.
    """
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.byte_predictor = nn.Linear(config.hidden_size, 256)
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, encoded_bytes, global_output, byte_mask=None, cross_mask=None):
        print("\n[LocalDecoder] - Input encoded_bytes shape:", encoded_bytes.shape)
        h = encoded_bytes
        positions = torch.arange(encoded_bytes.size(1), device=encoded_bytes.device)
        
        for idx, layer in enumerate(self.layers):
            print(f"[LocalDecoder] - Passing through DecoderLayer {idx}")
            h = layer(h, global_output, self_mask=byte_mask, cross_mask=cross_mask, positions=positions)
            print(f"[LocalDecoder] - DecoderLayer {idx} output shape:", h.shape)
            h = self.dropout(h)
        logits = self.byte_predictor(h)
        print("[LocalDecoder] - Final logits shape:", logits.shape)
        return logits


# =============================================================================
#                    MODELO DE ENTROPÍA (SIN USO DE CACHÉ)
# =============================================================================

class EntropyLM(nn.Module):
    """
    Modelo de entropía liviano optimizado con skip connections y gates aprendibles,
    SIN uso de cache.
    """
    def __init__(
        self, 
        hidden_size: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        context_size: int = 512,
        dropout: float = 0.1,
        learnable_dropout: float = 0.15,
        max_seq_length: int = 4096,
        window_size: int = 128
    ):
        super().__init__()

        self.context_size = context_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        
        # Embedding inicial
        self.byte_embedding = nn.Embedding(256, hidden_size)
        
        # Componente principal
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size),
            enable_nested_tensor=True
        )
        
        # Skip connections aprendibles
        self.skip_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, hidden_size))
            for _ in range(num_layers)
        ])
        self.skip_dropout = nn.Dropout(learnable_dropout)
        
        # Gates adaptativos
        self.gate_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Dropout(learnable_dropout),
                nn.Sigmoid()
            )
            for _ in range(num_layers)
        ])
        
        # Capa de salida con gate final
        self.output_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(learnable_dropout),
            nn.Sigmoid()
        )
        self.output = nn.Linear(hidden_size, 256)
        
        self.dropout = nn.Dropout(dropout)
        self.learnable_dropout = nn.Dropout(learnable_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Máscara causal pre-generada
        self._initialize_mask_cache(max_seq_length)

    def _initialize_mask_cache(self, max_seq_length: int):
        masks = torch.triu(
            torch.ones(max_seq_length, max_seq_length),
            diagonal=1
        ).bool()
        self.register_buffer('cached_masks', masks, persistent=False)
    
    def _apply_skip_connection(self, x: torch.Tensor, skip: torch.Tensor, layer_idx: int) -> torch.Tensor:
        weighted_skip = self.skip_dropout(self.skip_weights[layer_idx]) * skip
        gate_input = torch.cat([x, weighted_skip], dim=-1)
        gate = self.gate_layers[layer_idx](gate_input)
        out = x + self.learnable_dropout(gate * weighted_skip)
        return out

    def forward(
        self,
        input_bytes: torch.Tensor,
        return_probabilities: bool = False
    ) -> torch.Tensor:
        print("\n[EntropyLM] - Input shape:", input_bytes.shape)
        batch_size, seq_length = input_bytes.shape
        device = input_bytes.device
        
        # Embeddings
        x = self.byte_embedding(input_bytes)
        x = self.dropout(x)
        print("[EntropyLM] - After embedding shape:", x.shape)
        
        # Máscara causal
        mask = self.get_causal_mask(seq_length).to(device)
        
        # Procesar con autocast
        with torch.cuda.amp.autocast():
            initial_state = x
            x = self.encoder(
                src=x,
                mask=mask,
                is_causal=True
            )
            
            # Skip final
            final_gate_input = torch.cat([x, initial_state], dim=-1)
            final_gate = self.output_gate(final_gate_input)
            x = x + self.learnable_dropout(final_gate * initial_state)
            
            # Generar salida
            logits = self.output(x)
            probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
        
        if return_probabilities:
            result = probabilities
        else:
            result = self.compute_entropy(probabilities)
        print("[EntropyLM] - Output shape:", result.shape)
        return result

    def get_causal_mask(self, seq_length: int) -> torch.Tensor:
        if seq_length <= self.cached_masks.size(0):
            return self.cached_masks[:seq_length, :seq_length]
        else:
            return torch.triu(
                torch.ones(seq_length, seq_length),
                diagonal=1
            ).bool()

    def compute_entropy(
        self,
        probabilities: torch.Tensor,
        use_sliding_window: bool = True
    ) -> torch.Tensor:
        """
        Ajustado para secuencias más cortas que self.window_size:
        Si la secuencia es menor que window_size, no se hace unfold.
        """
        seq_len = probabilities.size(1)

        if use_sliding_window and seq_len >= self.window_size:
            windows = probabilities.unfold(
                1, self.window_size, max(1, self.window_size // 2)
            )
            
            window_entropies = []
            for window in windows.unbind(2):
                entropy = -torch.sum(
                    window * torch.log2(torch.clamp(window, min=1e-10)),
                    dim=-1
                )
                window_entropies.append(entropy)
            
            return torch.stack(window_entropies).mean(dim=0)
        
        # Si la secuencia es menor que window_size,
        # calcular entropía de forma directa en toda la secuencia.
        return -torch.sum(
            probabilities * torch.log2(torch.clamp(probabilities, min=1e-10)),
            dim=-1
        )


# =============================================================================
#                           BLT (Byte-Level Transformer)
# =============================================================================

class BLT(nn.Module):
    """
    Byte-Level Transformer (BLT) con parcheo adaptativo optimizado,
    SIN usar cache en ningún lugar.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.local_encoder = LocalEncoder(config)
        self.global_transformer = GlobalTransformer(config)
        self.local_decoder = LocalDecoder(config)
        
        self.entropy_model = EntropyLM(
            hidden_size=config.hidden_size,
            num_layers=config.entropy_model_layers,
            num_heads=config.num_heads,
            context_size=config.entropy_context_size,
            dropout=config.attention_dropout
        )
        
        self.global_norm = RMSNorm(config.hidden_size)
        self.global_dropout = nn.Dropout(config.resid_dropout)
        
        self.min_patch_size = config.min_patch_size
        self.max_patch_size = config.max_patch_size
        
        self.learnable_base_threshold = nn.Parameter(
            torch.tensor(config.initial_entropy_threshold)
        )
        self.learnable_std_scale = nn.Parameter(
            torch.tensor(0.5)
        )
        
        self.window_size = 128  
        self.stats_buffer = {}
        
        self.param_dropout = nn.Dropout(p=0.1)

    def _compute_adaptive_threshold(self, entropies: torch.Tensor) -> torch.Tensor:
        """
        Ajustado para secuencias cortas en unfold.
        """
        print("[BLT] - _compute_adaptive_threshold")
        batch_size = entropies.size(0)
        means = []
        stds = []
        
        for i in range(batch_size):
            seq_len = entropies[i].size(0)
            # Ajuste: si la secuencia es menor que self.window_size, no usar sliding
            if seq_len >= self.window_size:
                windows = entropies[i].unfold(0, self.window_size, max(1, self.window_size // 2))
                window_mean = windows.mean(dim=1, keepdim=True)
                window_std = torch.sqrt(
                    torch.var(windows, dim=1, keepdim=True, unbiased=False) + 1e-6
                )
                means.append(window_mean.mean())
                stds.append(window_std.mean())
            else:
                # Calcular media y std en toda la secuencia
                entire_mean = entropies[i].mean()
                entire_std = entropies[i].std(unbiased=False)
                means.append(entire_mean)
                stds.append(entire_std)

        mean = torch.stack(means).view(batch_size, 1)
        std = torch.stack(stds).view(batch_size, 1)
        
        base_threshold = self.param_dropout(self.learnable_base_threshold)
        std_scale = self.param_dropout(self.learnable_std_scale)
        
        threshold = torch.sigmoid(
            base_threshold + std_scale * std
        ).clamp(min=0.1, max=0.9)
        
        print("[BLT] - Adaptive threshold shape:", threshold.shape)
        return threshold

    def compute_patches(self, bytes_input: torch.Tensor) -> List[torch.Tensor]:
        print("[BLT] - compute_patches")
        batch_size = bytes_input.size(0)
        boundaries_list = []

        training_progress = getattr(self, 'current_step', 0) / getattr(self, 'total_steps', 1000) \
                            if self.training else 1.0
        
        # Fase inicial: uso de parches fijos mínimos
        if training_progress < 0.2:
            base_boundaries = torch.arange(
                self.min_patch_size,
                bytes_input.size(1),
                self.min_patch_size,
                device=bytes_input.device
            )
            boundaries_list = [base_boundaries for _ in range(batch_size)]
            return boundaries_list
        
        # Fase adaptativa
        with torch.cuda.amp.autocast():
            entropies = self.entropy_model(bytes_input)
            if self.training:
                entropies = F.dropout(entropies, p=0.1, training=True)
        
        thresholds = self._compute_adaptive_threshold(entropies)
        
        for i in range(batch_size):
            current_entropies = entropies[i]
            threshold = thresholds[i]
            
            boundaries = []
            current_size = 0
            last_boundary = 0
            
            for pos, entropy in enumerate(current_entropies):
                current_size = pos - last_boundary + 1
                should_split = any([
                    current_size >= self.max_patch_size,
                    current_size >= self.min_patch_size and entropy > threshold
                ])
                
                if should_split:
                    boundaries.append(pos)
                    last_boundary = pos + 1
            
            boundaries_tensor = torch.tensor(
                boundaries,
                device=bytes_input.device,
                dtype=torch.long
            )
            boundaries_list.append(boundaries_tensor)
        
        return boundaries_list

    def forward(self, bytes_input: torch.Tensor, patch_boundaries: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        print("\n[BLT] - Forward start")
        batch_size, seq_length = bytes_input.size()
        
        byte_encodings = self.local_encoder(bytes_input)
        byte_encodings = self.global_dropout(byte_encodings)
        
        if patch_boundaries is None:
            patch_boundaries = self.compute_patches(bytes_input)
        
        patch_means = []
        max_patches = 0
        
        for i in range(batch_size):
            boundaries = patch_boundaries[i]
            patches = []
            start = 0
            
            if boundaries.numel() > 0:
                if boundaries.dim() == 0:
                    boundaries = boundaries.unsqueeze(0)
                boundaries = torch.cat([
                    boundaries,
                    torch.tensor([seq_length], device=boundaries.device)
                ])
            else:
                boundaries = torch.tensor([seq_length], device=bytes_input.device)
            
            for end in boundaries:
                if end > start:
                    patch = byte_encodings[i, start:end].mean(dim=0)
                    patches.append(patch)
                    start = end
            
            if not patches:
                patches.append(byte_encodings[i].mean(dim=0))
            
            patches_tensor = torch.stack(patches)
            patch_means.append(patches_tensor)
            max_patches = max(max_patches, patches_tensor.size(0))
        
        padded_patches = torch.zeros(
            batch_size, max_patches, self.config.hidden_size,
            device=bytes_input.device,
            dtype=byte_encodings.dtype
        )
        
        for i, patches in enumerate(patch_means):
            num_patches = patches.size(0)
            padded_patches[i, :num_patches] = patches
        
        global_output = self.global_transformer(padded_patches)
        global_output = self.global_norm(global_output)
        global_output = self.global_dropout(global_output)
        
        logits = self.local_decoder(
            self.global_dropout(byte_encodings),
            global_output
        )
        print("[BLT] - Forward end, logits shape:", logits.shape)
        return logits


# =============================================================================
#                       CONFIGURACIÓN DEL MODELO BLT
# =============================================================================

class BLTConfig:
    """
    Configuración del Byte-Level Transformer (BLT).
    """
    def __init__(
        self,
        hidden_size=256,
        intermediate_size=1024,
        num_heads=4,
        encoder_layers=1,
        global_layers=6,
        decoder_layers=2,
        attention_dropout=0.12,
        resid_dropout=0.12,
        ngram_vocab_size=10000,
        window_size=512,
        max_position_embeddings=1024,
        entropy_model_layers=2,
        entropy_context_size=512,
        entropy_threshold=0.5,
        min_patch_size=32,
        max_patch_size=512,
        initial_entropy_threshold=0.5
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.global_layers = global_layers
        self.decoder_layers = decoder_layers
        self.attention_dropout = attention_dropout
        self.resid_dropout = resid_dropout
        self.ngram_vocab_size = ngram_vocab_size
        self.window_size = window_size
        self.max_position_embeddings = max_position_embeddings
        self.entropy_model_layers = entropy_model_layers
        self.entropy_context_size = entropy_context_size
        self.entropy_threshold = entropy_threshold
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.initial_entropy_threshold = initial_entropy_threshold


# =============================================================================
#                          UTILIDADES DE MÁSCARAS
# =============================================================================

def create_block_causal_mask(seq_length: int, window_size: int = None):
    """
    Crea una máscara causal con ventana opcional (sin cache).
    """
    mask = torch.ones((seq_length, seq_length), dtype=torch.bool)
    mask = torch.triu(mask, diagonal=1)
    
    if window_size:
        indices = torch.arange(seq_length)
        window_mask = (indices.unsqueeze(1) - indices.unsqueeze(0)).abs() <= window_size
        mask = mask | ~window_mask
    
    return mask

def create_patch_mask(patch_boundaries, seq_length):
    """
    Crea una máscara para parches con optimizaciones de memoria y soporte
    para parches dinámicos (sin cache).
    """
    if isinstance(patch_boundaries, list):
        patch_boundaries = torch.tensor(patch_boundaries)
    elif patch_boundaries.numel() == 0:
        return torch.zeros((seq_length, seq_length), dtype=torch.bool)
    
    if not torch.all(patch_boundaries[1:] > patch_boundaries[:-1]):
        patch_boundaries, _ = torch.sort(patch_boundaries)
    
    mask = torch.zeros((seq_length, seq_length), dtype=torch.bool)
    
    start = 0
    for end in patch_boundaries:
        if end > start:
            mask[start:end, start:end] = True
            start = end
    
    if start < seq_length:
        mask[start:, start:] = True
    
    return mask


# =============================================================================
#          CONFIGURACIÓN PARA PARCHEO (PatchingConfig) Y FUNCIONES AUX
# =============================================================================

class PatchingConfig:
    """
    Configuración para el esquema de parcheo.
    """
    def __init__(
        self,
        scheme='entropy',
        entropy_threshold=0.5,
        stride=4,
        reset_context=True,
        use_monotonic=True
    ):
        self.scheme = scheme
        self.entropy_threshold = entropy_threshold
        self.stride = stride
        self.reset_context = reset_context
        self.use_monotonic = use_monotonic


# =============================================================================
#                         FUNCIONES DE ENTRENAMIENTO
# =============================================================================

def train_step(model, optimizer, batch, patch_config):
    """
    Realiza un paso de entrenamiento, calculando la pérdida y aplicando backpropagation.
    """
    optimizer.zero_grad()

    input_bytes = batch[:, :-1]
    target_bytes = batch[:, 1:]
    patch_boundaries = None

    if patch_config.scheme == 'entropy':
        with torch.no_grad():
            entropies = model.entropy_model(input_bytes)
            indices = torch.where(entropies > patch_config.entropy_threshold)
            if indices[0].numel() == 0:
                patch_boundaries = torch.tensor([], dtype=torch.long, device=entropies.device)
            else:
                patch_boundaries = indices[1]

    elif patch_config.scheme == 'space':
        indices = torch.where(input_bytes == 32)
        if indices[0].numel() == 0:
            patch_boundaries = torch.tensor([], dtype=torch.long, device=input_bytes.device)
        else:
            patch_boundaries = indices[1] + 1

    elif patch_config.scheme == 'fixed':
        stride = patch_config.stride
        seq_length = input_bytes.size(1)
        patch_boundaries = torch.arange(
            stride, seq_length, stride, device=input_bytes.device
        )

    logits = model(input_bytes, patch_boundaries)

    logits_reshaped = logits.view(-1, logits.size(-1))
    target_bytes_reshaped = target_bytes.view(-1)
    loss = F.cross_entropy(logits_reshaped, target_bytes_reshaped)

    loss.backward()
    optimizer.step()

    return loss.item()


# =============================================================================
#                           FUNCIÓN DE GENERACIÓN
# =============================================================================

def generate(model, start_bytes, max_length=1000, temperature=1.0, top_k=20, patch_config=None, device='cpu'):
    """
    Genera una secuencia de bytes a partir de un contexto inicial.
    """
    model.eval()
    generated = list(start_bytes)
    
    with torch.no_grad():
        while len(generated) < max_length:
            input_bytes = torch.tensor(generated, device=device).unsqueeze(0)
            
            if patch_config is not None:
                if patch_config.scheme == 'entropy':
                    entropies = model.entropy_model(input_bytes)
                    if patch_config.use_monotonic:
                        entropy_diff = entropies[:, 1:] - entropies[:, :-1]
                        patch_boundaries = torch.where(entropy_diff > patch_config.entropy_threshold)[1] + 1
                    else:
                        patch_boundaries = torch.where(entropies > patch_config.entropy_threshold)[1]
                else:
                    patch_boundaries = torch.arange(
                        patch_config.stride, 
                        len(generated), 
                        patch_config.stride,
                        device=device
                    )
            else:
                patch_boundaries = None
            
            logits = model(input_bytes, patch_boundaries)

            # Forma: [1, seq_length, vocab_size]
            if logits.dim() == 3:
                logits = logits[0, -1] / temperature
            elif logits.dim() == 2:
                logits = logits[0] / temperature
            else:
                break
            
            top_k = min(top_k, logits.size(-1))
            topk_logits, topk_indices = torch.topk(logits, top_k)
            topk_probs = F.softmax(topk_logits, dim=-1)
            
            next_byte = topk_indices[torch.multinomial(topk_probs, 1).item()].item()
            generated.append(next_byte)
            
            if next_byte == 0:
                break
                
    return generated


# =============================================================================
#           CÁLCULO DE BITS POR BYTE (PERPLEJIDAD / ENTROPÍA)
# =============================================================================

def compute_bits_per_byte(model, data, patch_config):
    """
    Calcula bits por byte para un conjunto de datos dado, útil como métrica.
    """
    model.eval()
    total_loss = 0
    total_bytes = 0
    
    with torch.no_grad():
        for batch in data:
            input_bytes = batch[:-1]
            target_bytes = batch[1:]
            
            if patch_config.scheme == 'entropy':
                entropies = model.entropy_model(input_bytes.unsqueeze(0))
                patch_boundaries = torch.where(entropies > patch_config.entropy_threshold)[2]
            else:
                patch_boundaries = torch.arange(patch_config.stride, input_bytes.size(0), patch_config.stride)
            
            logits = model(input_bytes.unsqueeze(0), [patch_boundaries])
            loss = F.cross_entropy(logits.reshape(-1, 256), target_bytes.reshape(-1), reduction='sum')
            
            total_loss += loss.item()
            total_bytes += target_bytes.numel()
    
    bits_per_byte = total_loss / (total_bytes * math.log(2))
    return bits_per_byte
