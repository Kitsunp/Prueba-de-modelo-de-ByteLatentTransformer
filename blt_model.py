# blt_model_sin_cache_con_debug.py

import torch

# Habilitar Flash SDP explícitamente
torch.backends.cuda.enable_flash_sdp(enabled=True)

# Verificar si está habilitado
# # print(f"Flash SDP habilitado: {torch.backends.cuda.flash_sdp_enabled()}")

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

    Args:
        dim (int): Dimensión del vector de entrada a normalizar
        eps (float, opcional): Valor pequeño para evitar división por cero. Por defecto 1e-6

    Atributos:
        scale (float): Factor de escala precomputado basado en la dimensión
        g (nn.Parameter): Parámetro aprendible para reescalar la salida normalizada
        eps (float): Epsilon para estabilidad numérica
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))
        self.eps = eps
        
        # Inicialización con pequeño offset para evitar ceros exactos
        with torch.no_grad():
            self.g.add_(self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica normalización RMS al tensor de entrada.

        Args:
            x (torch.Tensor): Tensor de entrada de shape (..., dim)

        Returns:
            torch.Tensor: Tensor normalizado del mismo shape que la entrada

        Notas:
            - Añade epsilon tanto al numerador como al denominador para estabilidad
            - Realiza validaciones para detectar valores inválidos
        """

        # Añadir pequeño offset para evitar ceros exactos
        x = x + self.eps
        
        # Cálculo de norma con epsilon para estabilidad
        norm = torch.norm(x + self.eps, dim=-1, keepdim=True) * self.scale
        norm = norm + self.eps  # Prevenir división por cero
        
        # Normalización con validación
        out = (x / norm) * self.g
        

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
        # print("RotaryEmbedding - Positions shape:", positions.shape)
        positions = positions.unsqueeze(-1)
        freqs = positions.float() * self.inv_freq.unsqueeze(0)
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        return freqs_cos, freqs_sin

    def rotate_queries_and_keys(self, q, k, positions):
        """
        Aplica la rotación de RoPE a queries (q) y keys (k).
        """
        # print("RotaryEmbedding - rotate Q/K shapes:", q.shape, k.shape)
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

        # print("RotaryEmbedding - Rotated Q/K shapes:", q_rotate.shape, k_rotate.shape)
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
        # print("HeadwiseNorm - Input shape:", x.shape)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta
        # print("HeadwiseNorm - Output shape:", out.shape)
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
        # print("\n[MultiHeadAttention] - Input X shape:", x.shape)
        batch_size, seq_length, _ = x.size()
        x_norm = self.norm(x)

        # print("[MultiHeadAttention] - After RMSNorm X shape:", x_norm.shape)

        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        # print("[MultiHeadAttention] - Q/K/V projected shapes:", q.shape, k.shape, v.shape)

        def reshape_to_heads(tensor):
            return tensor.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape_to_heads(q), reshape_to_heads(k), reshape_to_heads(v)
        # print("[MultiHeadAttention] - Q/K/V reshaped to heads:", q.shape, k.shape, v.shape)

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
        # print("[MultiHeadAttention] - attn1/attn2 shapes:", attn1.shape, attn2.shape)

        attn1 = self.proj_dropout(attn1)
        attn2 = self.proj_dropout(attn2)

        # Calcular lambda para combinar
        lambda_val = self.compute_lambda()[:, :self.num_heads//2].unsqueeze(-1).unsqueeze(-1)
        out = attn1 - lambda_val * attn2

        # Concatenar cabezas
        out = torch.cat([out, out], dim=1)
        # print("[MultiHeadAttention] - Concat attn shape:", out.shape)

        # Normalización por cabeza y dropout posterior
        out = self.head_norm(out)
        out = self.resid_dropout(out)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        out = self.o_proj(out)
        out = self.resid_dropout(out)
        # print("[MultiHeadAttention] - Output shape:", out.shape)
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
        # print("\n[CrossAttention] - Input X shape:", x.shape)
        batch_size, seq_length, _ = x.size()
        context_length = context.size(1)

        x_norm = self.norm(x)
        # print("[CrossAttention] - After RMSNorm X shape:", x_norm.shape)

        q = self.q_proj(x_norm)
        k = self.k_proj(context)
        v = self.v_proj(context)

        # print("[CrossAttention] - Q/K/V shapes:", q.shape, k.shape, v.shape)
        q = self.proj_dropout(q)
        k = self.proj_dropout(k)
        v = self.proj_dropout(v)

        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)

        # print("[CrossAttention] - Q/K/V reshaped to heads:", q.shape, k.shape, v.shape)

        attn_mask = None
        if patch_mask is not None:
            attn_mask = (patch_mask == 0)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )
        # print("[CrossAttention] - Attn output shape:", out.shape)

        out = self.dropout(out)
        
        out = out.transpose(1, 2).reshape(batch_size, seq_length, self.hidden_size)
        out = self.o_proj(out)
        out = self.dropout(out)
        # print("[CrossAttention] - Output shape:", out.shape)
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
        # print("\n[FeedForward] - Input shape:", x.shape)
        x = self.norm(x)
        # print("[FeedForward] - After RMSNorm shape:", x.shape)
        swish = F.silu(self.w1(x))
        gate = self.w3(x)
        x = swish * gate
        x = self.activation_dropout(x)
        x = self.w2(x)
        x = self.dropout(x)
        # print("[FeedForward] - Output shape:", x.shape)
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
        # print("\n[EncoderLayer] - Input shape:", x.shape)
        h = x + self.self_attn(x, mask=self_mask, positions=positions, is_causal=False)
        # print("[EncoderLayer] - After Self-Attn shape:", h.shape)
        h = self.dropout(h)
        
        if cross_context is not None:
            h = h + self.cross_attn(h, cross_context, cross_mask)
            # print("[EncoderLayer] - After Cross-Attn shape:", h.shape)
            h = self.dropout(h)
        
        out = h + self.feed_forward(h)
        # print("[EncoderLayer] - After FeedForward shape:", out.shape)
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
        # print("\n[DecoderLayer] - Input shape:", x.shape)
        h = x + self.cross_attn(x, encoder_output, cross_mask)
        # print("[DecoderLayer] - After Cross-Attn shape:", h.shape)
        h = self.dropout(h)
        
        h = h + self.self_attn(h, self_mask, positions, is_causal=True)
        # print("[DecoderLayer] - After Self-Attn shape:", h.shape)
        h = self.dropout(h)
        
        out = h + self.feed_forward(h)
        # print("[DecoderLayer] - After FeedForward shape:", out.shape)
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
        # print("\n[ByteEmbedding] - Input shape:", bytes_input.shape)
        device = bytes_input.device
        batch_size, seq_length = bytes_input.shape

        embeds = self.byte_embeddings(bytes_input).float()
        embeds = self.dropout(embeds)
        # print("[ByteEmbedding] - After byte embedding shape:", embeds.shape)
        
        for n in range(3, 9):
            if seq_length >= n:
                ngram_hashes = self.compute_ngram_hash(bytes_input, n)
                ngram_embeds = self.ngram_hash_embeddings[n-3](ngram_hashes)
                
                expanded_embeds = torch.zeros_like(embeds)
                expanded_embeds[:, :seq_length - n + 1, :] += ngram_embeds / n
                embeds = embeds + expanded_embeds
        # print("[ByteEmbedding] - Output shape:", embeds.shape)
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
        # print("\n[LocalEncoder] - Input shape:", bytes_input.shape)
        h = self.byte_embeddings(bytes_input)
        # print("[LocalEncoder] - After ByteEmbedding shape:", h.shape)
        h = self.embedding_dropout(h)
        
        positions = torch.arange(bytes_input.size(1), device=bytes_input.device)
        
        for idx, layer in enumerate(self.layers):
            # print(f"[LocalEncoder] - Passing through EncoderLayer {idx}")
            h = layer(h, positions=positions)
            # print(f"[LocalEncoder] - EncoderLayer {idx} output shape:", h.shape)
            h = self.dropout(h)
        # print("[LocalEncoder] - Final output shape:", h.shape)
        return h
class GlobalTransformer(nn.Module):
    """
    Procesa la información a nivel de parches con atención global.
    Optimizado internamente para reducir redundancias y uso de VRAM,
    SIN añadir nuevas caches ni eliminar la lógica principal.
    Conserva la misma interfaz y atributos para compatibilidad.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.expansion_rate = getattr(config, 'expansion_rate', 2)
        
        # ------------------------------------------------------------
        #  Submódulos principales: (se mantienen igual para compatibilidad)
        # ------------------------------------------------------------
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.global_layers)
        ])
        
        # Sistemas de Dropout
        self.dropout = nn.Dropout(config.resid_dropout)
        self.adaptive_dropout = nn.Dropout(0.0855)
        self.gate_dropout = nn.Dropout(0.0855)
        self.mem_dropout = nn.Dropout(0.0855)
        self.skip_dropout = nn.Dropout(0.0855)
        
        # Normalizaciones
        self.layer_norms = nn.ModuleList([
            RMSNorm(config.hidden_size, eps=1e-6)
            for _ in range(config.global_layers)
        ])
        self.input_norm = RMSNorm(config.hidden_size, eps=1e-6)
        self.output_norm = RMSNorm(config.hidden_size, eps=1e-6)
        
        # Normalizaciones específicas
        self.pre_width_norm = RMSNorm(config.hidden_size, eps=1e-6)
        self.post_width_norm = (
            RMSNorm(config.n_states, eps=1e-6) if hasattr(config, 'n_states') else None
        )
        self.post_alpha_norm = (
            RMSNorm(config.n_states, eps=1e-6) if hasattr(config, 'n_states') else None
        )
        self.pre_gate_norm = RMSNorm(config.hidden_size, eps=1e-6)
        self.post_laurel_norm = RMSNorm(config.hidden_size, eps=1e-6)
        self.pre_memory_norm = RMSNorm(config.hidden_size, eps=1e-6)
        self.post_memory_norm = RMSNorm(config.hidden_size, eps=1e-6)
        self.post_combined_norm = RMSNorm(config.hidden_size, eps=1e-6)
        
        # Skip Gates
        self.skip_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Dropout(0.0855),
                nn.Sigmoid()
            ) for _ in range(config.global_layers)
        ])
        
        # LAUREL
        self.laurel_alphas = nn.Parameter(torch.ones(config.global_layers))
        self.laurel_g = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size)
            for _ in range(config.global_layers)
        ])
        
        # Pesos Adaptativos
        self.adaptive_weights = nn.Parameter(torch.ones(config.global_layers))
        
        # Hyper-Connections
        self.hyper_static_beta = nn.Parameter(torch.ones(self.expansion_rate))
        
        init_alpha0 = torch.zeros((config.global_layers, self.expansion_rate, 1))
        for i in range(config.global_layers):
            init_alpha0[i, i % self.expansion_rate, 0] = 1.0
        
        self.hyper_static_alpha = nn.Parameter(
            torch.cat([
                init_alpha0,
                torch.eye(self.expansion_rate).unsqueeze(0).repeat(config.global_layers, 1, 1)
            ], dim=2)
        )
        
        hidden_size = config.hidden_size
        self.hyper_dynamic_alpha_fn = nn.Parameter(
            torch.zeros((config.global_layers, hidden_size, self.expansion_rate + 1))
        )
        self.hyper_dynamic_alpha_scale = nn.Parameter(
            torch.ones(config.global_layers) * 0.01
        )
        self.hyper_dynamic_beta_fn = nn.Parameter(
            torch.zeros((config.global_layers, hidden_size))
        )
        self.hyper_dynamic_beta_scale = nn.Parameter(
            torch.ones(config.global_layers) * 0.01
        )
        
        # Memoria Jerárquica
        self.hierarchical_mem = nn.Parameter(
            torch.zeros(config.global_layers, config.hidden_size) + 1e-6
        )
        self.mem_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Dropout(0.085), 
            nn.Sigmoid()
        )

    # ----------------------------------------------------------------
    # Métodos internos para skip+laurel, hyper-connections y memoria
    # Se reorganizan para optimizar la implementación, sin añadir caches.
    # ----------------------------------------------------------------
    
    def _apply_skip_and_laurel(self, x, residual, layer_output, layer_idx):
        """
        Mezcla la salida de la capa (layer_output) con la entrada (x, residual)
        mediante gates adaptativos (skip_gates) y el mecanismo LAUREL.
        """
        # Normalizaciones fusionadas
        norm_x = self.layer_norms[layer_idx](x) * 0.1
        norm_r = self.layer_norms[layer_idx](residual) * 0.1
        
        gx = self.pre_gate_norm(norm_x) * 0.1
        gr = self.pre_gate_norm(norm_r) * 0.1
        
        # Gate input
        gate_input = torch.cat([gx, gr], dim=-1)  # (B, S, 2D)
        gate_val = self.skip_gates[layer_idx](gate_input)  # Sigmoid
        gate_val = self.gate_dropout(gate_val)
        
        # Pesos adaptativos
        aw = torch.sigmoid(self.adaptive_weights[layer_idx]) * 0.1
        aw = aw.view(1, 1, 1)
        weighted_r = aw * gate_val * norm_r
        weighted_r = self.skip_dropout(weighted_r)
        
        # LAUREL
        alpha = torch.sigmoid(self.laurel_alphas[layer_idx]) * 0.1
        alpha = self.adaptive_dropout(alpha)

        g_x = self.laurel_g[layer_idx](x) * 0.1
        g_x = self.skip_dropout(g_x)
        
        laurel_out = layer_output * alpha + g_x
        laurel_out = self.post_laurel_norm(laurel_out) * 0.1
        
        combined_output = laurel_out + weighted_r
        return combined_output

    def _apply_hyper_connections(self, x, hyper_h, layer_idx):
        """
        Mezcla la señal x con el tensor hyper_h a través de matrices alpha y beta,
        manteniendo la lógica original y sin añadir cache externo.
        """
        # Normalizar hyper_h
        norm_h = self.layer_norms[layer_idx](hyper_h)
        B, S, N, D = norm_h.shape  # N = n_states (expansion_rate)
        
        # Flatten + pre_width_norm
        h_flat = norm_h.reshape(B * S * N, D)
        h_flat = self.pre_width_norm(h_flat)
        
        # dynamic alpha
        alpha_fn = self.hyper_dynamic_alpha_fn[layer_idx][:, :N]  # (D, N)
        wc_weight = torch.matmul(h_flat, alpha_fn)
        
        if self.post_width_norm is not None:
            wc_weight = self.post_width_norm(wc_weight.reshape(-1, N))
        wc_weight = wc_weight.reshape(B, S, N, N)
        wc_weight = torch.tanh(wc_weight)
        
        alpha_scale = self.hyper_dynamic_alpha_scale[layer_idx].view(1, 1, 1, 1)
        dynamic_alpha = wc_weight * alpha_scale
        
        static_alpha = self.hyper_static_alpha[layer_idx][:N, :N]
        static_alpha = static_alpha.view(1, 1, N, N).expand(B, S, -1, -1)
        alpha = dynamic_alpha + static_alpha
        
        if self.post_alpha_norm is not None:
            alpha_view = alpha.reshape(-1, N)
            alpha_view = self.post_alpha_norm(alpha_view)
            alpha = alpha_view.reshape(B, S, N, N)
        
        # dynamic beta
        beta_fn = self.hyper_dynamic_beta_fn[layer_idx]  # (D,)
        dc_weight = torch.matmul(norm_h, beta_fn.view(-1, 1)).squeeze(-1)
        dc_weight = torch.tanh(dc_weight)
        
        beta_scale = self.hyper_dynamic_beta_scale[layer_idx].view(1, 1, 1)
        dynamic_beta = dc_weight * beta_scale
        
        static_beta = self.hyper_static_beta[:N].view(1, 1, -1)
        beta = dynamic_beta + static_beta
        
        # Mezcla final
        # alpha: (B,S,N,N), hyper_h: (B,S,N,D)
        mix_h = torch.matmul(alpha, hyper_h)
        
        x_expanded = x.unsqueeze(2).expand(-1, -1, N, -1)
        depth_conn = x_expanded * beta.unsqueeze(-1)
        
        return mix_h + depth_conn

    def _apply_hierarchical_memory(self, x, layer_idx, batch_size):
        """
        Integra la memoria jerárquica de la capa layer_idx, 
        manteniendo la lógica original sin añadir cachés.
        """
        # Sumar offset y reescalar
        x = (x + 1e-6) * 0.1
        x = self.pre_memory_norm(x) + 1e-6
        
        # Extraer y expandir la memoria para esta capa
        mem = self.hierarchical_mem[layer_idx:layer_idx+1] + 1e-6
        mem = mem.unsqueeze(0).expand(batch_size, x.size(1), -1)
        mem = self.mem_dropout(mem)
        
        # Calcular gate
        mem_input = torch.cat([x, mem], dim=-1)
        mem_gate_val = self.mem_gate(mem_input) * 0.1 + 1e-6
        
        memory_output = mem_gate_val * mem + 1e-6
        memory_output = self.post_memory_norm(memory_output) * 0.1 + 1e-6
        memory_output = self.mem_dropout(memory_output) * 0.1
        
        result = x + memory_output
        if torch.isnan(result).any():
            result = torch.nan_to_num(result, nan=1e-6)
        return result

    # ----------------------------------------------------------------
    #  Forward principal: igual firma y pasos, sin añadir caches extras
    # ----------------------------------------------------------------

    def forward(self, 
                patch_embeddings: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Mismo forward y firma, sin cache adicional.
        """
        batch_size = patch_embeddings.size(0)
        
        # Normalización de entrada + dropout
        h = self.input_norm(patch_embeddings)
        h = self.dropout(h)
        
        # Expansión "virtual" para hyper-connections
        hyper_h = h.unsqueeze(2).expand(-1, -1, self.expansion_rate, -1)
        positions = torch.arange(patch_embeddings.size(1), device=patch_embeddings.device)
        
        for idx, layer in enumerate(self.layers):
            # Paso por la capa
            prev_h = self.dropout(h)
            layer_out = layer(h, self_mask=attention_mask, positions=positions)
            
            # Combinación Skip & Laurel
            combined_out = self._apply_skip_and_laurel(prev_h, prev_h, layer_out, idx)
            
            # Hyper-connections
            hyper_h = self._apply_hyper_connections(combined_out, hyper_h, idx)
            
            # Suma con la media de hyper_h
            combined_features = combined_out + hyper_h.mean(dim=2)
            combined_features = self.post_combined_norm(combined_features)
            
            # Memoria jerárquica
            h = self._apply_hierarchical_memory(combined_features, idx, batch_size)
            h = self.dropout(h)
        
        # Normalización de salida + dropout
        h = self.output_norm(h)
        h = self.dropout(h)
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
        # print("\n[LocalDecoder] - Input encoded_bytes shape:", encoded_bytes.shape)
        h = encoded_bytes
        positions = torch.arange(encoded_bytes.size(1), device=encoded_bytes.device)
        
        for idx, layer in enumerate(self.layers):
            # print(f"[LocalDecoder] - Passing through DecoderLayer {idx}")
            h = layer(h, global_output, self_mask=byte_mask, cross_mask=cross_mask, positions=positions)
            # print(f"[LocalDecoder] - DecoderLayer {idx} output shape:", h.shape)
            h = self.dropout(h)
        logits = self.byte_predictor(h)
        # print("[LocalDecoder] - Final logits shape:", logits.shape)
        return logits


# =============================================================================
#                    MODELO DE ENTROPÍA (SIN USO DE CACHÉ)
# =============================================================================
class EntropyLM(nn.Module):
    """
    Modelo de lenguaje basado en entropía dual con análisis local y global.

    Este modelo implementa un sistema de entropía dual que:
    1. Analiza patrones locales en ventanas pequeñas.
    2. Captura contexto global en una dimensión reducida.
    3. Combina ambas medidas de forma adaptativa mediante pesos aprendidos.
    
    El modelo utiliza skip connections y gates aprendibles para mejorar
    la eficiencia y el flujo de información, sin necesidad de cache adicional.
    """

    # ========================================================
    #                SUBMÓDULOS INTERNOS
    # ========================================================
    class AdaptiveWaveletLayer(nn.Module):
        """
        Capa optimizada de análisis global usando wavelets neuronales adaptativos.
        Implementa procesamiento por lotes vectorizado sin uso de caché.
        """
        def __init__(self, hidden_size, global_size, num_wavelets=8, dropout=0.1, chunk_size=1024):
            super().__init__()
            self.hidden_size = hidden_size
            self.global_size = global_size
            self.num_wavelets = num_wavelets
            self.chunk_size = chunk_size
            
            # Wavelets optimizados para procesamiento por lotes
            self.mother_wavelets = nn.Parameter(
                torch.randn(1, num_wavelets, hidden_size, 1) * 0.02
            )
            
            # Escalas con broadcasting optimizado
            self.scales = nn.Parameter(torch.ones(1, num_wavelets, 1, 1))
            
            # Mixer optimizado con menos parámetros y mejor regularización
            self.coeff_mixer = nn.Sequential(
                nn.Linear(num_wavelets, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, hidden_size)
            )
            
            # Proyección con skip connection residual
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, global_size),
                nn.Dropout(dropout)
            )
            
            # MultiheadAttention optimizada
            self.num_heads = 4
            self.head_dim = hidden_size // self.num_heads
            self.scale = self.head_dim ** -0.5
            
            # Proyecciones para Q, K, V
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            
            self.attention_dropout = nn.Dropout(dropout)
                
        def _scaled_dot_product_attention(self, q, k, v, mask=None):
            """Implementación optimizada de atención sin caché."""
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                attn_weights = attn_weights.masked_fill(mask, float('-inf'))
                
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attention_dropout(attn_weights)
            
            return torch.matmul(attn_weights, v)
            
        def _attention_forward(self, x):
            """Forward pass de atención optimizado sin caché."""
            batch_size, seq_len, _ = x.shape
            
            # Proyectar Q, K, V y asegurar que sean contiguos
            q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).contiguous()
            k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).contiguous()
            v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).contiguous()
            
            # Transponer después de asegurar que son contiguos
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Generar máscara causal on-the-fly
            if self.training:
                mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1
                ).unsqueeze(0).unsqueeze(0)
            else:
                mask = None
                
            # Calcular atención
            attn_output = self._scaled_dot_product_attention(q, k, v, mask)
            
            # Reorganizar y proyectar salida
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            return self.out_proj(attn_output)
            
        def _compute_wavelet_coeffs_batched(self, x):
            """Calcula coeficientes wavelet de forma vectorizada para todo el batch."""
            batch_size, seq_len, channels = x.shape
            
            # Normalizar wavelets on-the-fly
            normalized_wavelets = F.normalize(self.mother_wavelets, dim=2)
            scaled_wavelets = normalized_wavelets * torch.sigmoid(self.scales)
            
            # Reorganizar para la convolución y asegurar que sea contiguo
            x_reshaped = x.reshape(batch_size * seq_len, 1, channels).contiguous()
            
            # Preparar wavelets para convolución
            wavelet_kernels = scaled_wavelets.squeeze(0).transpose(1, 2).contiguous()
            
            # Convolución vectorizada para todos los wavelets simultáneamente
            coeffs = F.conv1d(x_reshaped, wavelet_kernels, groups=1)
            
            return coeffs.reshape(batch_size, seq_len, self.num_wavelets)
        
        def _process_chunk(self, x_chunk):
            """Procesa un chunk de la secuencia."""
            # Calcular coeficientes wavelet
            coeffs = self._compute_wavelet_coeffs_batched(x_chunk)
            
            # Mezclar coeficientes
            mixed = self.coeff_mixer(coeffs)
            
            # Aplicar atención
            attn_output = self._attention_forward(mixed)
                
            return self.output_proj(attn_output + mixed)
        
        def forward(self, x):
            """Forward pass optimizado con procesamiento por chunks."""
            batch_size, seq_len, _ = x.shape
            
            # Para secuencias cortas, procesar directamente
            if seq_len <= self.chunk_size:
                return self._process_chunk(x)
            
            # Para secuencias largas, procesar por chunks
            outputs = []
            for start in range(0, seq_len, self.chunk_size):
                end = min(start + self.chunk_size, seq_len)
                chunk = x[:, start:end, :]
                chunk_output = self._process_chunk(chunk)
                outputs.append(chunk_output)
                
            # Concatenar resultados
            return torch.cat(outputs, dim=1)

    class CrossAttention(nn.Module):
        """
        Atención cruzada optimizada para interacción bidireccional entre características
        locales y globales. Mantiene la precisión en la detección de patrones mientras
        mejora la eficiencia computacional.
        """
        def __init__(self, hidden_size, global_size, num_heads, dropout):
            super().__init__()
            self.hidden_size = hidden_size
            self.global_size = global_size
            self.num_heads = num_heads
            self.dropout = dropout
            
            # Escalado para estabilidad numérica
            self.local_scale = (global_size // (num_heads // 2 if num_heads > 1 else 1)) ** -0.5
            self.global_scale = (hidden_size // num_heads) ** -0.5
            
            # Proyecciones optimizadas sin bias para reducir parámetros
            # pero manteniendo capacidad representativa
            self.local2global = nn.Linear(hidden_size, global_size, bias=True)
            self.global2local = nn.Linear(global_size, hidden_size, bias=True)
            
            # Proyecciones Q,K,V para atención local->global
            self.local_to_global_qkv = nn.ModuleDict({
                'q': nn.Linear(global_size, global_size, bias=True),
                'k': nn.Linear(global_size, global_size, bias=True),
                'v': nn.Linear(global_size, global_size, bias=True)
            })
            
            # Proyecciones Q,K,V para atención global->local
            self.global_to_local_qkv = nn.ModuleDict({
                'q': nn.Linear(hidden_size, hidden_size, bias=True),
                'k': nn.Linear(hidden_size, hidden_size, bias=True),
                'v': nn.Linear(hidden_size, hidden_size, bias=True)
            })
            
            # Proyecciones de salida
            self.local_out = nn.Linear(hidden_size, hidden_size, bias=False)
            self.global_out = nn.Linear(global_size, global_size, bias=False)
            
            # Dropouts estratégicos
            self.attn_dropout = nn.Dropout(dropout)
            self.proj_dropout = nn.Dropout(dropout)
        
        def _scaled_dot_product_attention(self, q, k, v, mask=None, scale=None):
            """
            Implementación optimizada de atención que usa flash attention cuando está disponible.
            """
            if hasattr(F, 'scaled_dot_product_attention'):
                # Usar Flash Attention si está disponible
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False
                )
            else:
                # Implementación estándar optimizada
                scores = torch.matmul(q, k.transpose(-2, -1)) * (scale or 1.0)
                
                if mask is not None:
                    scores = scores.masked_fill(mask, float('-inf'))
                
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.attn_dropout(attn_weights)
                attn_output = torch.matmul(attn_weights, v)
                
            return attn_output
        
        def _process_attention(self, q, k, v, num_heads, head_dim, mask=None, scale=None):
            """
            Procesa la atención manteniendo la precisión necesaria para detectar patrones.
            """
            batch_size, seq_len, _ = q.shape
            
            # Reshape preservando la información
            q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            
            # Calcular atención
            attn_output = self._scaled_dot_product_attention(q, k, v, mask, scale)
            
            # Reshape de vuelta
            return attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        
        def forward(self, local_feats, global_feats, local_mask=None, global_mask=None):
            """
            Forward pass optimizado que mantiene la precisión en la detección de patrones.
            
            Args:
                local_feats: [batch_size, seq_len, hidden_size]
                global_feats: [batch_size, seq_len, global_size]
                local_mask/global_mask: Máscaras opcionales para atención
            """
            batch_size, seq_len, _ = local_feats.shape
            
            # Proyección local->global
            local_as_global = self.local2global(local_feats)
            
            # Atención local->global
            q_global = self.local_to_global_qkv['q'](global_feats)
            k_global = self.local_to_global_qkv['k'](local_as_global)
            v_global = self.local_to_global_qkv['v'](local_as_global)
            
            global_head_dim = self.global_size // (self.num_heads // 2 if self.num_heads > 1 else 1)
            global_attn = self._process_attention(
                q_global, k_global, v_global,
                num_heads=self.num_heads // 2 if self.num_heads > 1 else 1,
                head_dim=global_head_dim,
                mask=local_mask,
                scale=self.local_scale
            )
            global_attn = self.global_out(global_attn)
            global_attn = self.proj_dropout(global_attn)
            
            # Proyección global->local
            global_as_local = self.global2local(global_feats)
            
            # Atención global->local
            q_local = self.global_to_local_qkv['q'](local_feats)
            k_local = self.global_to_local_qkv['k'](global_as_local)
            v_local = self.global_to_local_qkv['v'](global_as_local)
            
            local_head_dim = self.hidden_size // self.num_heads
            local_attn = self._process_attention(
                q_local, k_local, v_local,
                num_heads=self.num_heads,
                head_dim=local_head_dim,
                mask=global_mask,
                scale=self.global_scale
            )
            local_attn = self.local_out(local_attn)
            local_attn = self.proj_dropout(local_attn)
            
            # Actualización residual con dropout estratégico
            local_updated = local_feats + self.proj_dropout(local_attn)
            global_updated = global_feats + self.proj_dropout(global_attn)
            
            return local_updated, global_updated

    # ========================================================
    #                INICIALIZACIÓN DE LA CLASE
    # ========================================================

    def __init__(
        self, 
        hidden_size: int = 512,
        global_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        context_size: int = 512,
        dropout: float = 0.1,
        learnable_dropout: float = 0.12,
        max_seq_length: int = 4096,
        window_size: int = 128,
        vocab_size: int = 256  # Añadido para mayor generalidad
    ):
        super().__init__()

        self.context_size = context_size
        self.hidden_size = hidden_size
        self.global_size = global_size
        self.window_size = window_size
        self.vocab_size = vocab_size

        # ========== EMBEDDING ================
        self.byte_embedding = nn.Embedding(256, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        # ========== ANALISIS LOCAL =============
        local_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.local_encoder = nn.TransformerEncoder(
            encoder_layer=local_encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size),
            enable_nested_tensor=True
        )

        # ========== REDUCCION A GLOBAL (MEJORADA) ============
        self.global_reduction = self.AdaptiveWaveletLayer(
            hidden_size=hidden_size,
            global_size=global_size,
            num_wavelets=8,  # Ajustable según necesidades
            dropout=dropout
        )
        self.global_reduction_dropout = nn.Dropout(dropout)

        # ========== ANALISIS GLOBAL =============
        global_encoder_layer = nn.TransformerEncoderLayer(
            d_model=global_size,
            nhead=num_heads // 2 if num_heads > 1 else 1, 
            dim_feedforward=global_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.global_encoder = nn.TransformerEncoder(
            encoder_layer=global_encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(global_size),
            enable_nested_tensor=True
        )

        # ========== SKIP CONNECTIONS LOCALES =============
        self.local_skip_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, hidden_size))
            for _ in range(num_layers)
        ])
        self.local_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Dropout(learnable_dropout),
                nn.Sigmoid()
            )
            for _ in range(num_layers)
        ])

        # ========== SKIP CONNECTIONS GLOBALES =============
        self.global_skip_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, global_size))
            for _ in range(num_layers)
        ])
        self.global_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(global_size * 2, global_size),
                nn.Dropout(learnable_dropout),
                nn.Sigmoid()
            )
            for _ in range(num_layers)
        ])

        # ========== DROPOUT PARA SKIP CONNECTIONS =============
        self.skip_dropout = nn.Dropout(learnable_dropout)

        # ========== ATENCION CRUZADA =============
        self.cross_attention = self.CrossAttention(
            hidden_size=hidden_size,
            global_size=global_size,
            num_heads=num_heads,
            dropout=dropout
        )

        # ========== PONDERACION ADAPTATIVA =============
        self.weight_network = nn.Sequential(
            nn.Linear(hidden_size + global_size, 1),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )

        # ========== GATE FINAL Y SALIDA ============
        self.output_gate = nn.Sequential(
            nn.Linear((hidden_size + global_size) * 2, hidden_size + global_size),
            nn.Dropout(learnable_dropout),
            nn.Sigmoid()
        )
        self.output = nn.Linear(hidden_size + global_size, vocab_size)
        self.output_dropout = nn.Dropout(dropout)

        # ========== NORMALIZACIONES GENERALES =============
        self.dropout = nn.Dropout(dropout)
        self.learnable_dropout = nn.Dropout(learnable_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # ========== MASCARA CAUSAL PRE-GENERADA =============
        self._initialize_mask_cache(max_seq_length)
        self.base_smoothing = 0.1
        self.smoothing_scale = nn.Parameter(torch.tensor(0.5))
    # ========================================================
    #              METODOS INTERNOS
    # ========================================================
    def _compute_adaptive_smoothing(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Calcula un factor de suavizado adaptativo basado en la confianza del modelo
        y la temperatura dinámica del batch.

        Este método:
        1. Calcula un factor de suavizado base usando la probabilidad máxima
        2. Ajusta la temperatura según la entropía del batch
        3. Combina ambos efectos para suavizar adaptativamente las probabilidades

        Args:
            logits (torch.Tensor): Tensor de logits sin normalizar con shape [B, S, V]
                donde B es batch_size, S es sequence_length, y V es vocab_size.

        Returns:
            torch.Tensor: Factor de suavizado por posición con shape [B, S].
                        Los valores están entre [0, base_smoothing].
                        - Valores más altos indican más suavizado (alta confianza)
                        - Valores más bajos indican menos suavizado (baja confianza)

        Ejemplo:
            Para una secuencia donde el modelo está:
            - Muy seguro (p=0.9) → factor ≈ base_smoothing
            - Moderadamente seguro (p=0.6) → factor ≈ 0.5 * base_smoothing  
            - Inseguro (p=0.3) → factor ≈ 0.1 * base_smoothing
        """
        # 1. Obtener distribución de probabilidad base
        probs = F.softmax(logits, dim=-1)
        
        # 2. Calcular máxima probabilidad por posición [B, S]
        max_probs = probs.max(dim=-1)[0]
        
        # 3. Calcular entropía por posición y promedio del batch
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        mean_entropy = entropy.mean()
        
        # 4. Factor de temperatura basado en entropía
        temperature = torch.sigmoid(mean_entropy)
        temperature = temperature.clamp(0.1, 2.0)
        
        # 5. Calcular factor adaptativo basado en confianza
        confidence_factor = torch.sigmoid(self.smoothing_scale * (max_probs - 0.5))
        
        # 6. Combinar efectos
        # - Alta temperatura reduce el efecto del suavizado
        # - Baja temperatura mantiene el suavizado basado en confianza
        adaptive_factor = confidence_factor / temperature.unsqueeze(-1).expand_as(confidence_factor)
        
        # 7. Escalar al rango final y aplicar límites
        smoothing = (self.base_smoothing * adaptive_factor).clamp(0.0, self.base_smoothing)
        
        return smoothing

    def _initialize_mask_cache(self, max_seq_length: int):
        masks = torch.triu(
            torch.ones(max_seq_length, max_seq_length),
            diagonal=1
        ).bool()
        self.register_buffer('cached_masks', masks, persistent=False)

    def get_causal_mask(self, seq_length: int) -> torch.Tensor:
        if seq_length <= self.cached_masks.size(0):
            return self.cached_masks[:seq_length, :seq_length]
        return torch.triu(
            torch.ones(seq_length, seq_length),
            diagonal=1
        ).bool()

    def _apply_local_skip(self, x: torch.Tensor, skip: torch.Tensor, layer_idx: int) -> torch.Tensor:
        weighted_skip = self.skip_dropout(self.local_skip_weights[layer_idx]) * skip
        gate_input = torch.cat([x, weighted_skip], dim=-1)
        gate = self.local_gates[layer_idx](gate_input)
        out = x + self.learnable_dropout(gate * weighted_skip)
        return out

    def _apply_global_skip(self, x: torch.Tensor, skip: torch.Tensor, layer_idx: int) -> torch.Tensor:
        weighted_skip = self.skip_dropout(self.global_skip_weights[layer_idx]) * skip
        gate_input = torch.cat([x, weighted_skip], dim=-1)
        gate = self.global_gates[layer_idx](gate_input)
        out = x + self.learnable_dropout(gate * weighted_skip)
        return out

    # ========================================================
    #                FORWARD PRINCIPAL
    # ========================================================
    def forward(
        self,
        input_bytes: torch.Tensor,
        return_probabilities: bool = False
    ) -> torch.Tensor:
        """
        Procesa la entrada aplicando análisis local y global mediante atención cruzada,
        combinando características de diferentes niveles de abstracción.

        Este método implementa el forward pass completo del modelo que:
        1. Procesa los bytes de entrada mediante embeddings
        2. Aplica análisis local y global en paralelo
        3. Combina información mediante atención cruzada
        4. Utiliza skip connections y gates adaptativos
        5. Genera probabilidades o entropía según se requiera

        Args:
            input_bytes (torch.Tensor): Tensor de entrada con forma [batch_size, seq_length] 
                                    conteniendo índices de bytes.
            return_probabilities (bool, opcional): Si True, retorna distribución de probabilidades
                                                suavizada. Si False, retorna entropía.
                                                Por defecto: False.

        Returns:
            torch.Tensor: Si return_probabilities es True:
                        - Tensor de probabilidades suavizadas con forma [batch_size, seq_length, vocab_size]
                        Si return_probabilities es False:
                        - Tensor de entropía con forma [batch_size, seq_length] o [batch_size, window_size]

        Notas sobre el flujo de datos:
            1. Embeddings: input_bytes -> x [B, S, H]
            2. Análisis Local: x -> local_features [B, S, H]
            3. Análisis Global: x -> global_x -> global_features [B, S, G]
            4. Atención Cruzada: (local_features, global_features) -> (local_attn, global_attn)
            5. Skip Connections: Mejora flujo de gradientes
            6. Combinación Adaptativa: Mezcla características locales y globales
            7. Gate Final: Controla flujo de información residual
            8. Salida: Genera logits -> probabilidades/entropía

        Donde:
            B = batch_size
            S = sequence_length
            H = hidden_size
            G = global_size
        """
        # Asegurar que las dimensiones de entrada sean escalares
        batch_size, seq_length = input_bytes.shape
        seq_length = int(seq_length.item()) if isinstance(seq_length, torch.Tensor) else int(seq_length)
        device = input_bytes.device

        # ========= EMBEDDINGS INICIALES =========
        x = self.byte_embedding(input_bytes)
        x = self.embedding_dropout(x)

        # ========= MÁSCARA CAUSAL =========
        mask = self.get_causal_mask(seq_length).to(device)

        # Usamos autocast para mezclar precisión (solo si usamos GPUs con AMP)
        with torch.cuda.amp.autocast():
            # ========= ANALISIS LOCAL =========
            local_features = self.local_encoder(
                src=x,
                mask=mask,
                is_causal=True
            )

            # ========= ANALISIS GLOBAL =========
            # 1. Reducción MLP (captura no lineal)
            global_x = self.global_reduction(x)
            global_x = self.global_reduction_dropout(global_x)

            # 2. Encoder Global
            global_features = self.global_encoder(
                src=global_x,
                mask=mask,
                is_causal=True
            )

            # ========= ATENCIÓN CRUZADA (Local <-> Global) =========
            local_attn, global_attn = self.cross_attention(
                local_feats=local_features,
                global_feats=global_features,
                local_mask=mask,       # Opcional, enmascarado local
                global_mask=mask       # Opcional, enmascarado global
            )

            # ========= SKIP CONNECTIONS POR CAPA =========
            for layer_idx in range(len(self.local_skip_weights)):
                local_attn = self._apply_local_skip(local_attn, local_features, layer_idx)
                global_attn = self._apply_global_skip(global_attn, global_features, layer_idx)

            # ========= COMBINACIÓN ADAPTATIVA LOCAL-GLOBAL =========
            combined_features = torch.cat([local_attn, global_attn], dim=-1)
            weights = self.weight_network(combined_features)  # [B, S, 1]

            weighted_local = local_attn * weights
            weighted_global = global_attn * (1 - weights)
            final_features = torch.cat([weighted_local, weighted_global], dim=-1)

            # ========= CONCATENACION PARA GATE FINAL =========
            initial_state = torch.cat([x, global_x], dim=-1)
            final_gate_input = torch.cat([final_features, initial_state], dim=-1)
            final_gate = self.output_gate(final_gate_input)
            combined = final_features + self.learnable_dropout(final_gate * initial_state)

            # ========= DROPOUT Y SALIDA =========
            combined = self.output_dropout(combined)
            logits = self.output(combined)
            probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)

        if return_probabilities:
            probs = F.softmax(logits, dim=-1)
            # Aplicar suavizado adaptativo
            smoothing = self._compute_adaptive_smoothing(logits)
            vocab_size = logits.size(-1)
            smoothed_probs = probs * (1 - smoothing.unsqueeze(-1)) + smoothing.unsqueeze(-1) / vocab_size
            return smoothed_probs

        return self.compute_entropy(probabilities)

    # ========================================================
    #             CÁLCULO DE ENTROPÍA
    # ========================================================
    def compute_entropy(self, probabilities: torch.Tensor, use_sliding_window: bool = True) -> torch.Tensor:
        """
        Calcula la entropía H(P) = -Σ p(x)*log2 p(x).

        Si `use_sliding_window` es True y la longitud de la secuencia es mayor o igual a `window_size`,
        utiliza ventanas deslizantes para calcular la entropía de forma local y luego promedia los resultados.
        De lo contrario, calcula la entropía de manera global sobre toda la secuencia.

        Args:
            probabilities (torch.Tensor): Tensor de probabilidades con forma [B, S, V],
                                        donde
                                        B = tamaño del batch,
                                        S = longitud de la secuencia,
                                        V = número de categorías o variables.
            use_sliding_window (bool, opcional): Indica si se debe usar una ventana deslizante para secuencias largas.
                                                Por defecto es True.

        Returns:
            torch.Tensor: Tensor de entropía con forma [B, window_size] si se usa ventana deslizante,
                        o [B, S] si se calcula globalmente.
        """
        B, S, V = probabilities.shape
        seq_length = S.item() if isinstance(S, torch.Tensor) else S
        
        if use_sliding_window and seq_length >= self.window_size:
            # Generar ventanas deslizantes
            # La función `unfold` crea un nuevo tensor con ventanas deslizantes a lo largo de la dimensión de la secuencia.
            # Parámetros:
            #   dimension=1: dimensión de la secuencia.
            #   size=self.window_size: tamaño de cada ventana.
            #   step=max(1, self.window_size // 2): desplazamiento entre ventanas (mitad del tamaño de la ventana o al menos 1).
            # Resultado: [B, num_windows, window_size, V]
            windows = probabilities.unfold(1, self.window_size, max(1, self.window_size // 2))
            
            # Reorganizar dimensiones para facilitar el cálculo vectorizado de la entropía
            # Cambiar a [B, window_size, num_windows, V]
            windows = windows.permute(0, 2, 1, 3)

            # Cálculo vectorizado de la entropía por ventana
            # Aplicamos log2 a las probabilidades, asegurando que no haya valores menores que 1e-10 para evitar log(0)
            # Multiplicamos elemento a elemento por las probabilidades y sumamos sobre la última dimensión (V)
            # Resultado: [B, window_size, num_windows]
            ent = -torch.sum(
                windows * torch.log2(torch.clamp(windows, min=1e-10)),
                dim=-1
            )

            # Promediar la entropía sobre el número de ventanas (dim=2)
            # Resultado final: [B, window_size]
            return ent.mean(dim=2)
        else:
            # Cálculo directo de la entropía sobre toda la secuencia
            # Aplicamos log2 a las probabilidades, asegurando que no haya valores menores que 1e-10 para evitar log(0)
            # Multiplicamos elemento a elemento por las probabilidades y sumamos sobre la última dimensión (V)
            # Resultado: [B, S]
            ent = -torch.sum(
                probabilities * torch.log2(torch.clamp(probabilities, min=1e-10)),
                dim=-1
            )
            return ent
# =============================================================================
#                           BLT (Byte-Level Transformer)
# =============================================================================

class BLT(nn.Module):
    """
    Byte-Level Transformer (BLT) con parcheo adaptativo optimizado,
    SIN usar cache en ningún lugar.

    Esta versión mejorada integra explícitamente las probabilidades
    generadas por EntropyLM, con el fin de refinar la lógica de
    segmentación de parches y aprovechar la información de confiabilidad.
    """

    def __init__(self, config):
        """
        Args:
            config: Objeto de configuración (BLTConfig o similar) que contiene:
                - hidden_size: Dimensión oculta de los embeddings
                - resid_dropout: Tasa de dropout residual
                - attention_dropout: Tasa de dropout en la atención
                - min_patch_size: Tamaño mínimo de parche
                - max_patch_size: Tamaño máximo de parche
                - initial_entropy_threshold: Valor inicial del umbral de entropía
                - num_heads, etc.: Otros hiperparámetros relevantes
                - (Opcional) param prob_factor: Factor para combinar entropía y prob.
        """
        super().__init__()
        self.config = config
        
        # =================== Submódulos Principales ===================
        self.local_encoder = LocalEncoder(config)
        self.global_transformer = GlobalTransformer(config)
        self.local_decoder = LocalDecoder(config)
        
        # EntropyLM mejora: Se usará para obtener entropía y/o probabilidades
        self.entropy_model = EntropyLM(
            hidden_size=config.hidden_size,
            num_layers=config.entropy_model_layers,
            num_heads=config.num_heads,
            context_size=config.entropy_context_size,
            dropout=config.attention_dropout
        )
        
        # Normalización y Dropout
        self.global_norm = RMSNorm(config.hidden_size)
        self.global_dropout = nn.Dropout(config.resid_dropout)
        
        # Configuración de Parcheo
        self.min_patch_size = config.min_patch_size
        self.max_patch_size = config.max_patch_size
        
        # Parámetros Aprendibles para Umbrales
        # (Se utilizan para calcular el umbral adaptativo de entropía)
        self.learnable_base_threshold = nn.Parameter(
            torch.tensor(config.initial_entropy_threshold)
        )
        self.learnable_std_scale = nn.Parameter(
            torch.tensor(0.5)
        )
        
        # Parámetro adicional para ponderar el uso de probabilidades
        # Ajusta cuánto peso se le da a la "discrepancia" de prob. frente a la entropía
        # (Podrías exponerlo como config.prob_factor si lo deseas)
        self.prob_factor = getattr(config, "prob_factor", 0.3)
        
        # Tamaño de la ventana para cálculos de estadística adaptativa
        self.window_size = 128
        self.stats_buffer = {}
        
        # Dropout para parámetros
        self.param_dropout = nn.Dropout(p=0.1)
        
        # Atributos para rastrear el progreso de entrenamiento
        self.current_step = 0
        self.total_steps = 1000

    # ------------------------------------------------------------------
    #         Funciones Internas para Cálculo del Umbral Adaptativo
    # ------------------------------------------------------------------
    def _compute_adaptive_threshold(self, entropies: torch.Tensor) -> torch.Tensor:
        """
        Ajusta dinámicamente el umbral de entropía basándose en estadísticos
        (media y desviación estándar). Soporta secuencias cortas.

        Args:
            entropies (torch.Tensor): Tensor [B, S] con las entropías.
        Returns:
            threshold (torch.Tensor): Umbrales por cada muestra del batch [B, 1].
        """
        batch_size = entropies.size(0)
        means = []
        stds = []
        
        for i in range(batch_size):
            seq_len = entropies[i].size(0)
            # Si la secuencia >= window_size, usar ventanas deslizantes
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
        
        # Aplicar sigmoide y limitar en [0.1, 0.9]
        threshold = torch.sigmoid(base_threshold + std_scale * std).clamp(min=0.1, max=0.9)
        return threshold

    # ------------------------------------------------------------------
    #          Función Auxiliar para Integrar Prob y Entropía
    # ------------------------------------------------------------------
    def _compute_boundary_score(
        self,
        entropies: torch.Tensor,
        probabilities: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula un vector de puntajes de corte, combinando la entropía
        y la discrepancia entre probabilidades consecutivas.

        Args:
            entropies: [S] Entropías de la secuencia (una por posición).
            probabilities: [S, V=256] Probabilidades por cada posición.

        Returns:
            scores: [S-1] Puntaje para cada posición j, j in [0..S-2],
                    indicando cuán favorable es hacer un corte entre j y j+1.
        """
        seq_len = probabilities.size(0)
        if seq_len < 2:
            # Si la secuencia es muy corta, no calculamos "discrepancia"
            return entropies[:-1]  # Devolvemos algo sencillo

        # Discrepancia de prob.: Distancia L1 entre P_j e P_{j+1}
        # shape: [S-1]
        prob_diff = torch.sum(
            torch.abs(probabilities[:-1] - probabilities[1:]),
            dim=-1
        )

        # Por simplicidad, el puntaje de corte en la posición j se define como:
        # score_j = ent_j + prob_factor * prob_diff_j
        # Donde ent_j = entropía en la posición j, y prob_diff_j indica
        # el "salto" en la distribución de prob. entre j y j+1.
        # Podríamos refinar con ent_{j+1} también, etc.
        # Ajustar a gusto la mezcla.
        cut_scores = entropies[:-1] + (self.prob_factor * prob_diff)

        return cut_scores

    # ------------------------------------------------------------------
    #             Cálculo de Parques con Entropía + Probabilidades
    # ------------------------------------------------------------------
    def compute_patches(self, bytes_input: torch.Tensor) -> List[torch.Tensor]:
        """
        Calcula las fronteras (boundaries) de los parches en la secuencia de bytes de entrada
        utilizando una combinación adaptativa de entropía y discrepancia en las probabilidades.

        Esta versión mejorada integra tanto la entropía como las probabilidades generadas
        por el modelo `EntropyLM` para decidir de manera adaptativa dónde segmentar los parches.
        Además, asegura que todas las condiciones booleanas se manejen correctamente para
        evitar errores como "Boolean value of Tensor with more than one value is ambiguous".

        Args:
            bytes_input (torch.Tensor): Tensor de entrada con forma [batch_size, seq_length],
                                        conteniendo índices de bytes.

        Returns:
            List[torch.Tensor]: Lista de tensores, cada uno conteniendo las posiciones de
                                corte (boundaries) para cada muestra en el batch.
        """
        batch_size = bytes_input.size(0)
        boundaries_list = []

        # ====================== Progreso de Entrenamiento ======================
        # Calcula el progreso de entrenamiento para ajustar la mezcla de boundaries fijos y adaptativos.
        training_progress = (
            getattr(self, 'current_step', 0) / getattr(self, 'total_steps', 1000)
        ) if self.training else 1.0

        # Determina el factor de mezcla basado en el progreso de entrenamiento.
        if training_progress < 0.2:
            mix_factor = 0.0  # Fase inicial: completamente fija
        elif training_progress > 0.8:
            mix_factor = 1.0  # Fase final: completamente adaptativa
        else:
            # Fase intermedia: mezcla gradual entre fija y adaptativa
            raw_mix = (training_progress - 0.2) / 0.6
            mix_factor = 1 / (1 + torch.exp(torch.tensor(-10 * (raw_mix - 0.5))))

        # ====================== Boundaries Fijos Base ======================
        # Calcula los boundaries fijos basados en el tamaño mínimo de parche.
        seq_len = bytes_input.size(1)
        if self.min_patch_size <= seq_len:
            base_boundaries = torch.arange(
                self.min_patch_size,
                seq_len,
                self.min_patch_size,
                device=bytes_input.device
            )
        else:
            base_boundaries = torch.tensor([], device=bytes_input.device, dtype=torch.long)

        # ====================== Fase Completamente Fija ======================
        # Si el mix_factor es 0, retorna únicamente los boundaries fijos.
        if mix_factor == 0:
            return [base_boundaries for _ in range(batch_size)]

        # ====================== Cálculo de Entropía y Probabilidades ======================
        with torch.cuda.amp.autocast():
            # 1. Obtiene las probabilidades del modelo EntropyLM
            probabilities = self.entropy_model(bytes_input, return_probabilities=True)  # [B, S, V]

            # 2. Obtiene las entropías del modelo EntropyLM
            entropies = self.entropy_model(bytes_input, return_probabilities=False)  # [B, S]

            # 3. Aplica dropout a las entropías si está en modo de entrenamiento
            if self.training:
                entropies = F.dropout(entropies, p=0.1, training=True)

        # ====================== Cálculo de Umbrales Adaptativos ======================
        # Calcula umbrales de entropía adaptativos para cada posición en la secuencia.
        thresholds = self._compute_adaptive_threshold(entropies)  # [B, 1]

        # ====================== Procesamiento por Cada Muestra del Batch ======================
        for b_idx in range(batch_size):
            current_ent = entropies[b_idx]         # [S]
            current_prob = probabilities[b_idx]    # [S, V]
            threshold = thresholds[b_idx].item()    # Valor escalar

            # Combina entropía y discrepancias de probabilidad para calcular scores de corte.
            # shape: [S-1]
            boundary_scores = self._compute_boundary_score(current_ent, current_prob)

            adaptive_boundaries = []
            last_boundary = 0
            S = current_ent.size(0)

            # Itera sobre cada posición en la secuencia para decidir si cortar.
            for pos in range(S):
                current_size = pos - last_boundary + 1

                if pos < S - 1:
                    # Verifica las condiciones de corte:
                    # 1) Tamaño del parche >= max_patch_size
                    # 2) Tamaño del parche >= min_patch_size y score de corte > umbral adaptativo
                    condition1 = current_size >= self.max_patch_size
                    condition2 = (
                        (current_size >= self.min_patch_size) and
                        (boundary_scores[pos].item() > threshold * (1 + mix_factor))
                    )
                    cut_condition = condition1 or condition2
                else:
                    # Forzar el corte al final de la secuencia.
                    cut_condition = (pos == S - 1)

                # Si se cumple la condición de corte, agregar la posición como boundary.
                if cut_condition:
                    adaptive_boundaries.append(pos)
                    last_boundary = pos + 1

            # Convierte la lista de boundaries adaptativos a un tensor.
            adaptive_boundaries = torch.tensor(
                adaptive_boundaries, device=bytes_input.device, dtype=torch.long
            )

            # ====================== Mezcla de Boundaries Fijos y Adaptativos ======================
            if mix_factor == 1:
                # Si mix_factor es 1, usa solo boundaries adaptativos.
                final_boundaries = adaptive_boundaries
            else:
                # Combina boundaries fijos y adaptativos ponderados por mix_factor.
                fixed_weight = 1 - mix_factor
                adaptive_weight = mix_factor
                num_boundaries = int(
                    fixed_weight * len(base_boundaries) +
                    adaptive_weight * len(adaptive_boundaries)
                )

                if num_boundaries == 0:
                    # Si no hay boundaries, retorna un tensor vacío.
                    final_boundaries = torch.tensor([], device=bytes_input.device, dtype=torch.long)
                else:
                    # Combina los boundaries fijos y adaptativos.
                    if len(base_boundaries) > 0 and len(adaptive_boundaries) > 0:
                        combined = torch.cat([
                            base_boundaries * fixed_weight,
                            adaptive_boundaries.float() * adaptive_weight  # Convierte a float para la multiplicación
                        ])
                    elif len(base_boundaries) > 0:
                        combined = base_boundaries.float() * fixed_weight
                    else:
                        combined = adaptive_boundaries.float() * adaptive_weight

                    # Selecciona los top boundaries según num_boundaries.
                    if combined.numel() > num_boundaries:
                        _, indices = torch.topk(torch.abs(combined), num_boundaries)
                        final_boundaries = torch.sort(combined[indices])[0].long()
                    else:
                        final_boundaries = torch.sort(combined)[0].long()

            # ====================== Asegurar Límites y Limpieza ======================
            # Filtra boundaries que estén dentro de los límites permitidos.
            final_boundaries = final_boundaries[
                (final_boundaries >= self.min_patch_size) &
                (final_boundaries <= seq_len - 1)
            ]

            # Agrega los boundaries finales al listado.
            boundaries_list.append(final_boundaries)

        return boundaries_list


    # ------------------------------------------------------------------
    #                 Forward Principal del BLT
    # ------------------------------------------------------------------
    def forward(
        self,
        bytes_input: torch.Tensor,
        patch_boundaries: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward principal:
          1) Cálculo de byte_encodings vía LocalEncoder
          2) Segmentación en parches (adaptativa) y reducción (mean)
          3) Paso por GlobalTransformer
          4) Decodificación (LocalDecoder) para obtener logits finales

        Args:
            bytes_input: [B, S] Tensor con bytes de entrada
            patch_boundaries: Lista opcional de boundaries para forzar parches

        Returns:
            logits: [B, S, 256] Logits sobre bytes.
        """
        batch_size, seq_length = bytes_input.size()

        # -------------------- Encodings Locales --------------------
        byte_encodings = self.local_encoder(bytes_input)
        byte_encodings = self.global_dropout(byte_encodings)

        # -------------------- Boundaries Adaptativos --------------------
        if patch_boundaries is None:
            patch_boundaries = self.compute_patches(bytes_input)

        # -------------------- Calcular Patches --------------------
        patch_means = []
        max_patches = 0

        for i in range(batch_size):
            boundaries = patch_boundaries[i]
            patches = []
            start = 0
            
            # Asegurar que boundaries tenga un final en seq_length
            if boundaries.numel() > 0:
                if boundaries.dim() == 0:
                    boundaries = boundaries.unsqueeze(0)
                boundaries = torch.cat([
                    boundaries,
                    torch.tensor([seq_length], device=boundaries.device)
                ])
            else:
                boundaries = torch.tensor([seq_length], device=bytes_input.device)
            
            # Reducir cada parche
            for end in boundaries:
                if end > start:
                    patch = byte_encodings[i, start:end].mean(dim=0)
                    patches.append(patch)
                    start = end
            
            # Si no hay parches creados, usar toda la secuencia
            if not patches:
                patches.append(byte_encodings[i].mean(dim=0))
            
            patches_tensor = torch.stack(patches)
            patch_means.append(patches_tensor)
            max_patches = max(max_patches, patches_tensor.size(0))
        
        # -------------------- Padding de Parches --------------------
        padded_patches = torch.zeros(
            batch_size, max_patches, self.config.hidden_size,
            device=bytes_input.device,
            dtype=byte_encodings.dtype
        )
        
        for i, patches in enumerate(patch_means):
            num_patches = patches.size(0)
            padded_patches[i, :num_patches] = patches

        # -------------------- Global Transformer --------------------
        global_output = self.global_transformer(padded_patches)
        global_output = self.global_norm(global_output)
        global_output = self.global_dropout(global_output)

        # -------------------- Decodificación Local --------------------
        logits = self.local_decoder(
            self.global_dropout(byte_encodings),
            global_output
        )
        return logits

    # ------------------------------------------------------------------
    #                 Utilidad: Progreso de Entrenamiento
    # ------------------------------------------------------------------
    def update_training_progress(self, current_step, total_steps):
        """
        Actualiza el progreso del entrenamiento para ajustar gradualmente
        la mezcla de boundaries fijos y adaptativos.
        """
        self.current_step = current_step
        self.total_steps = total_steps

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
        attention_dropout=0.1,
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
