# blt_model_sin_cache_con_debug.py

import torch

# Habilitar Flash SDP explícitamente para optimizar operaciones de atención
torch.backends.cuda.enable_flash_sdp(enabled=True)

# Verificar si Flash SDP está habilitado
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
    Normalización RMS (Root Mean Square) utilizada como alternativa a LayerNorm.
    Escala la norma RMS de cada vector a 1, proporcionando estabilidad durante el entrenamiento.

    Args:
        dim (int): Dimensionalidad de la entrada.

    Attributes:
        scale (float): Factor de escala basado en la dimensión, calculado como sqrt(dim).
        g (nn.Parameter): Parámetro de escala aprendido para reescalar la salida.
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5  # Factor de escala basado en la dimensión
        self.g = nn.Parameter(torch.ones(dim))  # Parámetro de escala aprendido

    def forward(self, x):
        """
        Aplica la normalización RMS a la entrada.

        Args:
            x (torch.Tensor): Tensor de entrada de forma [..., dim].

        Returns:
            torch.Tensor: Tensor normalizado con la misma forma que la entrada.
        """
        print("RMSNorm - Input shape:", x.shape)
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale  # Cálculo de la norma RMS
        out = x / norm * self.g  # Normalización y reescalado
        print("RMSNorm - Output shape:", out.shape)
        return out


class RotaryEmbedding(nn.Module):
    """
    Implementación de Rotary Embeddings para inyectar información posicional
    en las consultas y claves de la atención, mejorando la capacidad del modelo
    para capturar relaciones posicionales en los datos.

    Args:
        dim (int): Dimensionalidad de las embeddings.
        theta (float, opcional): Parámetro de frecuencia inicial. Por defecto es 500000.

    Attributes:
        dim (int): Dimensionalidad de las embeddings.
        theta (float): Parámetro de frecuencia para escalamiento de posiciones.
        inv_freq (torch.Tensor): Frecuencias inversas precomputadas para las rotaciones.
    """
    def __init__(self, dim, theta=500000):
        super().__init__()
        self.dim = dim  # Dimensionalidad de las embeddings
        self.theta = theta  # Parámetro de frecuencia
        # Calcular las frecuencias inversas para cada dimensión par
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)  # Almacenar como buffer para evitar entrenamiento

    def forward(self, positions):
        """
        Calcula los cosenos y senos correspondientes a las frecuencias rotatorias
        para las posiciones dadas.

        Args:
            positions (torch.Tensor): Tensor de posiciones de forma [seq_length].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensors de cosenos y senos con formas [seq_length, dim//2].
        """
        print("RotaryEmbedding - Positions shape:", positions.shape)
        positions = positions.unsqueeze(-1)  # Expandir dimensiones para el cálculo
        freqs = positions.float() * self.inv_freq.unsqueeze(0)  # Aplicar frecuencias inversas
        freqs_cos = torch.cos(freqs)  # Cálculo de cosenos
        freqs_sin = torch.sin(freqs)  # Cálculo de senos
        return freqs_cos, freqs_sin  # Retornar cosenos y senos

    def rotate_queries_and_keys(self, q, k, positions):
        """
        Aplica la rotación de Rotary Positional Embeddings (RoPE) a queries (q) y keys (k).

        Args:
            q (torch.Tensor): Tensor de queries de forma [batch_size, num_heads, seq_length, head_dim].
            k (torch.Tensor): Tensor de keys de forma [batch_size, num_heads, seq_length, head_dim].
            positions (torch.Tensor): Tensor de posiciones de forma [seq_length].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensors de queries y keys rotados con las mismas formas que las entradas.
        """
        print("RotaryEmbedding - rotate Q/K shapes:", q.shape, k.shape)
        freqs_cos, freqs_sin = self.forward(positions)  # Obtener cosenos y senos para las posiciones

        batch_size, num_heads, seq_length, head_dim = q.shape  # Extraer dimensiones
        dim_half = head_dim // 2  # Dividir la dimensión de la cabeza para la rotación

        freqs_cos = freqs_cos[:seq_length, :dim_half]  # Ajustar tamaño de cosenos
        freqs_sin = freqs_sin[:seq_length, :dim_half]  # Ajustar tamaño de senos
        
        freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(0)  # Añadir dimensiones de batch y cabezas
        freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(0)  # Añadir dimensiones de batch y cabezas

        # Dividir las dimensiones de head_dim en dos para la rotación
        q1, q2 = q[..., :dim_half], q[..., dim_half:]
        k1, k2 = k[..., :dim_half], k[..., dim_half:]

        # Aplicar la rotación a queries y keys
        q_rotate = torch.cat([
            q1 * freqs_cos - q2 * freqs_sin,  # Rotación para la primera mitad
            q2 * freqs_cos + q1 * freqs_sin   # Rotación para la segunda mitad
        ], dim=-1)

        k_rotate = torch.cat([
            k1 * freqs_cos - k2 * freqs_sin,  # Rotación para la primera mitad
            k2 * freqs_cos + k1 * freqs_sin   # Rotación para la segunda mitad
        ], dim=-1)

        print("RotaryEmbedding - Rotated Q/K shapes:", q_rotate.shape, k_rotate.shape)
        return q_rotate, k_rotate  # Retornar queries y keys rotados


class HeadwiseNorm(nn.Module):
    """
    Normalización específica por cabeza para atención multi-cabeza.
    Normaliza cada cabeza de atención de forma independiente, mejorando la estabilidad del entrenamiento
    y la representatividad de cada cabeza.

    Args:
        num_heads (int): Número de cabezas de atención.
        head_dim (int): Dimensionalidad de cada cabeza.
        eps (float, opcional): Valor de epsilon para estabilidad numérica. Por defecto es 1e-5.

    Attributes:
        num_heads (int): Número de cabezas de atención.
        head_dim (int): Dimensionalidad de cada cabeza de atención.
        eps (float): Valor de epsilon para estabilidad numérica.
        gamma (nn.Parameter): Parámetro de escala aprendido para cada cabeza.
        beta (nn.Parameter): Parámetro de desplazamiento aprendido para cada cabeza.
    """
    def __init__(self, num_heads, head_dim, eps=1e-5):
        super().__init__()
        self.num_heads = num_heads  # Número de cabezas de atención
        self.head_dim = head_dim  # Dimensionalidad de cada cabeza
        self.eps = eps  # Valor de epsilon para estabilidad numérica
        self.gamma = nn.Parameter(torch.ones(num_heads, 1, 1))  # Parámetro de escala aprendido
        self.beta = nn.Parameter(torch.zeros(num_heads, 1, 1))  # Parámetro de desplazamiento aprendido

    def forward(self, x):
        """
        Aplica la normalización específica por cabeza a la entrada.

        Args:
            x (torch.Tensor): Tensor de entrada de forma [batch_size, num_heads, seq_length, head_dim].

        Returns:
            torch.Tensor: Tensor normalizado con la misma forma que la entrada.
        """
        print("HeadwiseNorm - Input shape:", x.shape)
        mean = x.mean(dim=-1, keepdim=True)  # Calcular la media por dimensión de cabeza
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # Calcular la varianza por dimensión de cabeza
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # Normalizar
        out = self.gamma * x_norm + self.beta  # Reescalar y desplazar
        print("HeadwiseNorm - Output shape:", out.shape)
        return out


# =============================================================================
#                         ATENCIÓN MULTI-CABEZA
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention con múltiples niveles de dropout y Rotary Embeddings.
    Incorpora un mecanismo de combinación lambda para fusionar dos grupos de cabezas.

    Args:
        config (BLTConfig): Objeto de configuración que contiene los parámetros necesarios para la atención.

    Attributes:
        num_heads (int): Número de cabezas de atención.
        hidden_size (int): Dimensionalidad total de las capas de atención.
        head_dim (int): Dimensionalidad de cada cabeza de atención.
        q_proj (nn.Linear): Capa de proyección para queries.
        k_proj (nn.Linear): Capa de proyección para keys.
        v_proj (nn.Linear): Capa de proyección para values.
        o_proj (nn.Linear): Capa de proyección de salida.
        rotary (RotaryEmbedding): Módulo de Rotary Embeddings para inyectar información posicional.
        norm (RMSNorm): Capa de normalización RMS aplicada a la entrada.
        head_norm (HeadwiseNorm): Capa de normalización específica por cabeza después de la atención.
        attention_dropout (float): Tasa de dropout para la atención.
        resid_dropout (nn.Dropout): Capa de dropout para las conexiones residuales.
        proj_dropout (nn.Dropout): Capa de dropout para las proyecciones.
        lambda_dropout (nn.Dropout): Capa de dropout para los parámetros lambda.
        lambda_init (nn.Parameter): Parámetro lambda inicial.
        lambda_q1 (nn.Parameter): Parámetro lambda para el primer grupo de queries.
        lambda_k1 (nn.Parameter): Parámetro lambda para el primer grupo de keys.
        lambda_q2 (nn.Parameter): Parámetro lambda para el segundo grupo de queries.
        lambda_k2 (nn.Parameter): Parámetro lambda para el segundo grupo de keys.
    """
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads  # Número de cabezas de atención
        self.hidden_size = config.hidden_size  # Dimensionalidad total de la atención
        self.head_dim = self.hidden_size // self.num_heads  # Dimensionalidad de cada cabeza

        # Validaciones para asegurar consistencia en las dimensiones
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"  # RoPE requiere head_dim par
        assert self.head_dim * self.num_heads == self.hidden_size, "hidden_size must be divisible by num_heads"

        # Proyecciones lineales para queries, keys, y values
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)  # Proyección de queries
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)  # Proyección de keys
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)  # Proyección de values
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)  # Proyección de salida

        # Módulos adicionales
        self.rotary = RotaryEmbedding(self.head_dim)  # Embeddings rotatorios para posicionalidad
        self.norm = RMSNorm(config.hidden_size)  # Normalización RMS aplicada a la entrada
        self.head_norm = HeadwiseNorm(num_heads=self.num_heads, head_dim=self.head_dim)  # Normalización por cabeza

        # Parámetros de dropout
        self.attention_dropout = config.attention_dropout  # Dropout para la atención
        self.resid_dropout = nn.Dropout(config.resid_dropout)  # Dropout para conexiones residuales
        self.proj_dropout = nn.Dropout(config.resid_dropout)  # Dropout para las proyecciones
        self.lambda_dropout = nn.Dropout(self.attention_dropout)  # Dropout para parámetros lambda

        # Inicializar parámetros relacionados con lambda
        self._initialize_lambda_parameters(config)

    def _initialize_lambda_parameters(self, config):
        """
        Inicializa los parámetros lambda utilizados para combinar dos grupos de cabezas de atención.

        Args:
            config (BLTConfig): Objeto de configuración que contiene los parámetros necesarios.
        """
        layer_idx = getattr(config, 'layer_idx', 1)  # Obtener índice de capa, por defecto 1
        base_lambda = 0.8 - 0.6 * math.exp(-0.3 * (layer_idx - 1))  # Calcular lambda base
        self.lambda_init = nn.Parameter(torch.full((1, self.num_heads), base_lambda))  # Parámetro lambda inicial

        dim_scale = 0.01 / math.sqrt(self.hidden_size)  # Escala para inicialización
        # Parámetros lambda para los grupos de queries y keys
        self.lambda_q1 = nn.Parameter(torch.randn(1, self.num_heads, self.head_dim) * dim_scale)
        self.lambda_k1 = nn.Parameter(torch.randn(1, self.num_heads, self.head_dim) * dim_scale)
        self.lambda_q2 = nn.Parameter(torch.randn(1, self.num_heads, self.head_dim) * dim_scale)
        self.lambda_k2 = nn.Parameter(torch.randn(1, self.num_heads, self.head_dim) * dim_scale)

    def compute_lambda(self):
        """
        Calcula el valor lambda para combinar los dos grupos de cabezas de atención.

        Returns:
            torch.Tensor: Tensor lambda de forma [1, num_heads].
        """
        # Aplicar dropout a los parámetros lambda y calcular productos internos
        qk1 = torch.sum(self.lambda_dropout(self.lambda_q1) * self.lambda_k1, dim=-1)  # [1, num_heads]
        qk2 = torch.sum(self.lambda_dropout(self.lambda_q2) * self.lambda_k2, dim=-1)  # [1, num_heads]
        # Combinar los resultados y añadir lambda_init
        lambda_val = torch.exp(qk1) - torch.exp(qk2) + self.lambda_init
        return torch.clamp(lambda_val, min=0.0, max=1.0)  # Limitar lambda entre 0 y 1

    def forward(self, x, mask=None, positions=None, is_causal=False):
        """
        Realiza el paso hacia adelante de la atención multi-cabeza.

        Args:
            x (torch.Tensor): Entrada de forma [batch_size, seq_length, hidden_size].
            mask (Optional[torch.Tensor], optional): Máscara de atención de forma [batch_size, seq_length]. Por defecto es None.
            positions (Optional[torch.Tensor], optional): Tensor de posiciones de forma [seq_length]. Por defecto es None.
            is_causal (bool, optional): Indica si la atención es causal (autoregresiva). Por defecto es False.

        Returns:
            torch.Tensor: Salida de la atención multi-cabeza de forma [batch_size, seq_length, hidden_size].
        """
        print("\n[MultiHeadAttention] - Input X shape:", x.shape)
        batch_size, seq_length, _ = x.size()
        x_norm = self.norm(x)  # Aplicar normalización RMS a la entrada

        print("[MultiHeadAttention] - After RMSNorm X shape:", x_norm.shape)

        q = self.q_proj(x_norm)  # Proyectar queries
        k = self.k_proj(x_norm)  # Proyectar keys
        v = self.v_proj(x_norm)  # Proyectar values
        print("[MultiHeadAttention] - Q/K/V projected shapes:", q.shape, k.shape, v.shape)

        def reshape_to_heads(tensor):
            """
            Reorganiza el tensor para separar las cabezas de atención.

            Args:
                tensor (torch.Tensor): Tensor de forma [batch_size, seq_length, hidden_size].

            Returns:
                torch.Tensor: Tensor de forma [batch_size, num_heads, seq_length, head_dim].
            """
            return tensor.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Reorganizar queries, keys y values para las cabezas
        q, k, v = reshape_to_heads(q), reshape_to_heads(k), reshape_to_heads(v)
        print("[MultiHeadAttention] - Q/K/V reshaped to heads:", q.shape, k.shape, v.shape)

        # Dividir las cabezas en dos grupos para el mecanismo lambda
        q1, q2 = torch.chunk(q, 2, dim=1)  # Dividir queries en dos grupos
        k1, k2 = torch.chunk(k, 2, dim=1)  # Dividir keys en dos grupos
        v1, v2 = torch.chunk(v, 2, dim=1)  # Dividir values en dos grupos

        # Aplicar Rotary Positional Embeddings a ambos grupos de queries y keys
        q1, k1 = self.rotary.rotate_queries_and_keys(q1, k1, positions)
        q2, k2 = self.rotary.rotate_queries_and_keys(q2, k2, positions)

        # Aplicar dropout después de RoPE
        q1, k1 = self.proj_dropout(q1), self.proj_dropout(k1)
        q2, k2 = self.proj_dropout(q2), self.proj_dropout(k2)

        if mask is not None:
            # Convertir mask a boolean y aplicar dropout
            mask = self.resid_dropout(mask.float()).bool()

        # Atención para el primer grupo de cabezas
        attn1 = F.scaled_dot_product_attention(
            q1, k1, v1, dropout_p=self.attention_dropout, is_causal=is_causal
        )
        # Atención para el segundo grupo de cabezas
        attn2 = F.scaled_dot_product_attention(
            q2, k2, v2, dropout_p=self.attention_dropout, is_causal=is_causal
        )
        print("[MultiHeadAttention] - attn1/attn2 shapes:", attn1.shape, attn2.shape)

        # Aplicar dropout a las salidas de atención
        attn1 = self.proj_dropout(attn1)
        attn2 = self.proj_dropout(attn2)

        # Calcular lambda para combinar los dos grupos de atención
        lambda_val = self.compute_lambda()[:, :self.num_heads//2].unsqueeze(-1).unsqueeze(-1)  # [1, num_heads//2, 1, 1]
        out = attn1 - lambda_val * attn2  # Combinar las atenciones con lambda

        # Concatenar las cabezas para obtener la forma original
        out = torch.cat([out, out], dim=1)  # [batch_size, num_heads, seq_length, head_dim]
        print("[MultiHeadAttention] - Concat attn shape:", out.shape)

        # Aplicar normalización específica por cabeza y dropout posterior
        out = self.head_norm(out)
        out = self.resid_dropout(out)

        # Reorganizar de vuelta a la forma original
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        out = self.o_proj(out)  # Proyección de salida
        out = self.resid_dropout(out)  # Aplicar dropout
        print("[MultiHeadAttention] - Output shape:", out.shape)
        return out  # Retornar la salida de atención multi-cabeza


class CrossAttention(nn.Module):
    """
    Atención cruzada (CrossAttention) para mezclar el contexto externo (context)
    con la entrada actual (x). Permite que el modelo incorpore información de
    fuentes externas al procesar la secuencia.

    Args:
        config (BLTConfig): Objeto de configuración que contiene los parámetros necesarios para la atención cruzada.

    Attributes:
        num_heads (int): Número de cabezas de atención.
        hidden_size (int): Dimensionalidad total de las capas de atención.
        head_dim (int): Dimensionalidad de cada cabeza de atención.
        q_proj (nn.Linear): Capa de proyección para queries.
        k_proj (nn.Linear): Capa de proyección para keys.
        v_proj (nn.Linear): Capa de proyección para values.
        o_proj (nn.Linear): Capa de proyección de salida.
        norm (RMSNorm): Capa de normalización RMS aplicada a la entrada.
        dropout (nn.Dropout): Capa de dropout para la atención.
        proj_dropout (nn.Dropout): Capa de dropout para las proyecciones de salida.
    """
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads  # Número de cabezas de atención
        self.hidden_size = config.hidden_size  # Dimensionalidad total de la atención
        self.head_dim = self.hidden_size // self.num_heads  # Dimensionalidad de cada cabeza

        # Proyecciones lineales para queries, keys, y values del contexto
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)  # Proyección de queries
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)  # Proyección de keys
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)  # Proyección de values
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)  # Proyección de salida
        
        self.norm = RMSNorm(config.hidden_size)  # Normalización RMS aplicada a la entrada
        self.dropout = nn.Dropout(config.attention_dropout)  # Dropout para la atención

        self.proj_dropout = nn.Dropout(config.resid_dropout)  # Dropout para las proyecciones

    def forward(self, x, context, patch_mask=None):
        """
        Realiza el paso hacia adelante de la atención cruzada.

        Args:
            x (torch.Tensor): Entrada de forma [batch_size, seq_length, hidden_size].
            context (torch.Tensor): Contexto externo de forma [batch_size, context_length, hidden_size].
            patch_mask (Optional[torch.Tensor], optional): Máscara para parches. Por defecto es None.

        Returns:
            torch.Tensor: Salida de la atención cruzada de forma [batch_size, seq_length, hidden_size].
        """
        print("\n[CrossAttention] - Input X shape:", x.shape)
        batch_size, seq_length, _ = x.size()
        context_length = context.size(1)  # Longitud del contexto

        x_norm = self.norm(x)  # Aplicar normalización RMS a la entrada
        print("[CrossAttention] - After RMSNorm X shape:", x_norm.shape)

        q = self.q_proj(x_norm)  # Proyectar queries
        k = self.k_proj(context)  # Proyectar keys desde el contexto
        v = self.v_proj(context)  # Proyectar values desde el contexto

        print("[CrossAttention] - Q/K/V shapes:", q.shape, k.shape, v.shape)
        q = self.proj_dropout(q)  # Aplicar dropout a queries
        k = self.proj_dropout(k)  # Aplicar dropout a keys
        v = self.proj_dropout(v)  # Aplicar dropout a values

        # Reorganizar queries, keys y values para las cabezas
        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        k = k.reshape(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, context_length, head_dim]
        v = v.reshape(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, context_length, head_dim]

        print("[CrossAttention] - Q/K/V reshaped to heads:", q.shape, k.shape, v.shape)

        attn_mask = None
        if patch_mask is not None:
            attn_mask = (patch_mask == 0)  # Crear máscara de atención si se proporciona patch_mask

        # Aplicar atención escalada por producto punto
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,  # Aplicar dropout solo en entrenamiento
            is_causal=False  # Atención no causal
        )
        print("[CrossAttention] - Attn output shape:", out.shape)

        out = self.dropout(out)  # Aplicar dropout a la salida de atención
        
        # Reorganizar de vuelta a la forma original
        out = out.transpose(1, 2).reshape(batch_size, seq_length, self.hidden_size)  # [batch_size, seq_length, hidden_size]
        out = self.o_proj(out)  # Proyección de salida
        out = self.dropout(out)  # Aplicar dropout
        print("[CrossAttention] - Output shape:", out.shape)
        return out  # Retornar la salida de atención cruzada


# =============================================================================
#                         FEEDFORWARD Y CAPAS DE ENCODER/DECODER
# =============================================================================

class FeedForward(nn.Module):
    """
    Capa FeedForward con activación SwiGLU y múltiples dropouts.
    Implementa una red feedforward que utiliza una puerta basada en SiLU (Swish) para mejorar la no linealidad.

    Args:
        config (BLTConfig): Objeto de configuración que contiene los parámetros necesarios para la capa feedforward.

    Attributes:
        w1 (nn.Linear): Primera capa lineal que expande la dimensionalidad.
        w2 (nn.Linear): Segunda capa lineal que reduce la dimensionalidad a la original.
        w3 (nn.Linear): Capa lineal para la puerta de activación.
        norm (RMSNorm): Capa de normalización RMS aplicada a la entrada.
        dropout (nn.Dropout): Capa de dropout aplicada después de la segunda capa lineal.
        activation_dropout (nn.Dropout): Capa de dropout aplicada después de la activación SwiGLU.
    """
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size)  # Capa de expansión
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size)  # Capa de reducción
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size)  # Capa para la puerta de activación
        
        self.norm = RMSNorm(config.hidden_size)  # Normalización RMS aplicada a la entrada
        self.dropout = nn.Dropout(config.resid_dropout)  # Dropout aplicado después de w2
        self.activation_dropout = nn.Dropout(config.resid_dropout)  # Dropout aplicado después de la activación

    def forward(self, x):
        """
        Aplica la capa feedforward con activación SwiGLU.

        Args:
            x (torch.Tensor): Entrada de forma [batch_size, seq_length, hidden_size].

        Returns:
            torch.Tensor: Salida de la capa feedforward con la misma forma que la entrada.
        """
        print("\n[FeedForward] - Input shape:", x.shape)
        x = self.norm(x)  # Aplicar normalización RMS a la entrada
        print("[FeedForward] - After RMSNorm shape:", x.shape)
        swish = F.silu(self.w1(x))  # Aplicar activación SiLU (Swish) a la salida de w1
        gate = self.w3(x)  # Cálculo de la puerta de activación
        x = swish * gate  # Aplicar la puerta para controlar la activación
        x = self.activation_dropout(x)  # Aplicar dropout a la activación
        x = self.w2(x)  # Aplicar la segunda capa lineal
        x = self.dropout(x)  # Aplicar dropout a la salida de w2
        print("[FeedForward] - Output shape:", x.shape)
        return x  # Retornar la salida de la capa feedforward


class EncoderLayer(nn.Module):
    """
    Capa de encoder que combina self-attention, cross-attention (opcional)
    y feed-forward. Utiliza conexiones residuales y dropout para estabilizar el entrenamiento.

    Args:
        config (BLTConfig): Objeto de configuración que contiene los parámetros necesarios para la capa de encoder.

    Attributes:
        self_attn (MultiHeadAttention): Módulo de self-attention.
        cross_attn (CrossAttention): Módulo de cross-attention.
        feed_forward (FeedForward): Módulo feedforward.
        dropout (nn.Dropout): Capa de dropout aplicada después de cada subcapa.
    """
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)  # Self-attention para la entrada
        self.cross_attn = CrossAttention(config)  # Cross-attention opcional con contexto externo
        self.feed_forward = FeedForward(config)  # Capa feedforward
        self.dropout = nn.Dropout(config.resid_dropout)  # Dropout para conexiones residuales

    def forward(self, x, cross_context=None, self_mask=None, cross_mask=None, positions=None):
        """
        Aplica la capa de encoder con self-attention, cross-attention (si se proporciona contexto)
        y feedforward, utilizando conexiones residuales y dropout.

        Args:
            x (torch.Tensor): Entrada de forma [batch_size, seq_length, hidden_size].
            cross_context (Optional[torch.Tensor], optional): Contexto externo para cross-attention. Por defecto es None.
            self_mask (Optional[torch.Tensor], optional): Máscara para self-attention. Por defecto es None.
            cross_mask (Optional[torch.Tensor], optional): Máscara para cross-attention. Por defecto es None.
            positions (Optional[torch.Tensor], optional): Tensor de posiciones de forma [seq_length]. Por defecto es None.

        Returns:
            torch.Tensor: Salida de la capa de encoder con la misma forma que la entrada.
        """
        print("\n[EncoderLayer] - Input shape:", x.shape)
        # Aplicar self-attention con conexión residual
        h = x + self.self_attn(x, mask=self_mask, positions=positions, is_causal=False)
        print("[EncoderLayer] - After Self-Attn shape:", h.shape)
        h = self.dropout(h)  # Aplicar dropout después de la conexión residual
        
        if cross_context is not None:
            # Si se proporciona contexto, aplicar cross-attention con conexión residual
            h = h + self.cross_attn(h, cross_context, cross_mask)
            print("[EncoderLayer] - After Cross-Attn shape:", h.shape)
            h = self.dropout(h)  # Aplicar dropout después de la conexión residual
        
        # Aplicar feedforward con conexión residual
        out = h + self.feed_forward(h)
        print("[EncoderLayer] - After FeedForward shape:", out.shape)
        out = self.dropout(out)  # Aplicar dropout después de la conexión residual
        return out  # Retornar la salida de la capa de encoder


class DecoderLayer(nn.Module):
    """
    Capa de decoder con cross-attention, self-attention con enmascaramiento causal
    y feed-forward. Permite que el decoder incorpore información del encoder mientras mantiene la autoregresión en la generación.

    Args:
        config (BLTConfig): Objeto de configuración que contiene los parámetros necesarios para la capa de decoder.

    Attributes:
        cross_attn (CrossAttention): Módulo de cross-attention.
        self_attn (MultiHeadAttention): Módulo de self-attention con enmascaramiento causal.
        feed_forward (FeedForward): Módulo feedforward.
        dropout (nn.Dropout): Capa de dropout aplicada después de cada subcapa.
    """
    def __init__(self, config):
        super().__init__()
        self.cross_attn = CrossAttention(config)  # Cross-attention con el encoder
        self.self_attn = MultiHeadAttention(config)  # Self-attention con enmascaramiento causal
        self.feed_forward = FeedForward(config)  # Capa feedforward
        self.dropout = nn.Dropout(config.resid_dropout)  # Dropout para conexiones residuales

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None, positions=None):
        """
        Aplica la capa de decoder con cross-attention, self-attention causal y feedforward,
        utilizando conexiones residuales y dropout.

        Args:
            x (torch.Tensor): Entrada de forma [batch_size, seq_length, hidden_size].
            encoder_output (torch.Tensor): Salida del encoder de forma [batch_size, context_length, hidden_size].
            self_mask (Optional[torch.Tensor], optional): Máscara para self-attention. Por defecto es None.
            cross_mask (Optional[torch.Tensor], optional): Máscara para cross-attention. Por defecto es None.
            positions (Optional[torch.Tensor], optional): Tensor de posiciones de forma [seq_length]. Por defecto es None.

        Returns:
            torch.Tensor: Salida de la capa de decoder con la misma forma que la entrada.
        """
        print("\n[DecoderLayer] - Input shape:", x.shape)
        # Aplicar cross-attention con conexión residual
        h = x + self.cross_attn(x, encoder_output, cross_mask)
        print("[DecoderLayer] - After Cross-Attn shape:", h.shape)
        h = self.dropout(h)  # Aplicar dropout después de la conexión residual
        
        # Aplicar self-attention con enmascaramiento causal y conexión residual
        h = h + self.self_attn(h, self_mask, positions, is_causal=True)
        print("[DecoderLayer] - After Self-Attn shape:", h.shape)
        h = self.dropout(h)  # Aplicar dropout después de la conexión residual
        
        # Aplicar feedforward con conexión residual
        out = h + self.feed_forward(h)
        print("[DecoderLayer] - After FeedForward shape:", out.shape)
        out = self.dropout(out)  # Aplicar dropout después de la conexión residual
        return out  # Retornar la salida de la capa de decoder


# =============================================================================
#                          EMBEDDINGS A NIVEL DE BYTE
# =============================================================================

class ByteEmbedding(nn.Module):
    """
    Genera embeddings a nivel de byte e incluye n-gram embeddings para capturar patrones locales.
    Utiliza una función hash para mapear n-gramas a índices de embeddings, permitiendo manejar
    un vocabulario grande de n-gramas de manera eficiente.

    Args:
        config (BLTConfig): Objeto de configuración que contiene los parámetros necesarios para los embeddings.

    Attributes:
        byte_embeddings (nn.Embedding): Capa de embedding para bytes individuales.
        ngram_hash_embeddings (nn.ModuleList): Lista de capas de embedding para n-gramas de tamaños 3 a 8.
        dropout (nn.Dropout): Capa de dropout aplicada a los embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.byte_embeddings = nn.Embedding(256, config.hidden_size)  # Embeddings para bytes individuales
        self.ngram_hash_embeddings = nn.ModuleList([
            nn.Embedding(config.ngram_vocab_size, config.hidden_size)
            for _ in range(6)  # Para n-gramas de tamaño 3 a 8
        ])
        
        self.dropout = nn.Dropout(config.resid_dropout)  # Dropout aplicado a los embeddings

    def compute_ngram_hash(self, bytes_sequence, n):
        """
        Calcula el hash de n-gramas para mapearlos a índices de embeddings.

        Args:
            bytes_sequence (torch.Tensor): Secuencia de bytes de forma [batch_size, seq_length].
            n (int): Tamaño del n-grama.

        Returns:
            torch.Tensor: Tensor de índices de embeddings para los n-gramas de forma [batch_size, num_ngrams].
        """
        device = bytes_sequence.device  # Obtener el dispositivo
        batch_size, seq_length = bytes_sequence.shape  # Obtener dimensiones del batch

        if seq_length < n:
            return torch.empty((batch_size, 0), dtype=torch.long, device=device)  # Retornar vacío si seq_length < n

        ngrams = bytes_sequence.unfold(dimension=1, size=n, step=1)  # Extraer n-gramas
        exponents = torch.arange(n, device=device).float()  # Crear exponente para cada posición en el n-grama
        weights = (256 ** exponents).unsqueeze(0).unsqueeze(0)  # Calcular pesos para hash

        # Calcular el valor hash de cada n-grama
        hash_values = (ngrams.float() * weights).sum(dim=-1).long()  # Sumar ponderaciones
        hash_tensor = hash_values % self.ngram_hash_embeddings[n-3].num_embeddings  # Modulo para limitar el rango

        return hash_tensor  # Retornar índices de embeddings para n-gramas

    def forward(self, bytes_input):
        """
        Genera los embeddings a nivel de byte y añade los embeddings de n-gramas.

        Args:
            bytes_input (torch.Tensor): Tensor de entrada de bytes de forma [batch_size, seq_length].

        Returns:
            torch.Tensor: Tensor de embeddings enriquecidos de forma [batch_size, seq_length, hidden_size].
        """
        print("\n[ByteEmbedding] - Input shape:", bytes_input.shape)
        device = bytes_input.device  # Obtener el dispositivo
        batch_size, seq_length = bytes_input.shape  # Obtener dimensiones del batch

        # Generar embeddings para bytes individuales
        embeds = self.byte_embeddings(bytes_input).float()
        embeds = self.dropout(embeds)  # Aplicar dropout a los embeddings
        print("[ByteEmbedding] - After byte embedding shape:", embeds.shape)
        
        # Iterar sobre tamaños de n-gramas de 3 a 8
        for n in range(3, 9):
            if seq_length >= n:
                ngram_hashes = self.compute_ngram_hash(bytes_input, n)  # Calcular hashes de n-gramas
                ngram_embeds = self.ngram_hash_embeddings[n-3](ngram_hashes)  # Obtener embeddings para n-gramas
                
                # Crear un tensor de zeros para agregar los embeddings de n-gramas
                expanded_embeds = torch.zeros_like(embeds)
                # Asignar los embeddings de n-gramas a las posiciones correspondientes, ponderados por 1/n
                expanded_embeds[:, :seq_length - n + 1, :] += ngram_embeds / n
                embeds = embeds + expanded_embeds  # Sumar los embeddings de n-gramas al embedding base
        print("[ByteEmbedding] - Output shape:", embeds.shape)
        return embeds  # Retornar los embeddings enriquecidos


# =============================================================================
#                        MODELOS DE ENCODER Y DECODER
# =============================================================================

class LocalEncoder(nn.Module):
    """
    Encoder local que procesa los bytes de forma detallada mediante múltiples capas de encoder.
    Utiliza embeddings a nivel de byte y aplica dropout para regularización.

    Args:
        config (BLTConfig): Objeto de configuración que contiene los parámetros necesarios para el encoder local.

    Attributes:
        byte_embeddings (ByteEmbedding): Módulo de embeddings a nivel de byte.
        embedding_dropout (nn.Dropout): Capa de dropout aplicada a los embeddings.
        layers (nn.ModuleList): Lista de capas de encoder locales.
        dropout (nn.Dropout): Capa de dropout aplicada después de cada capa de encoder.
    """
    def __init__(self, config):
        super().__init__()
        self.byte_embeddings = ByteEmbedding(config)  # Embeddings a nivel de byte
        self.embedding_dropout = nn.Dropout(config.resid_dropout)  # Dropout aplicado a los embeddings
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])  # Capas de encoder locales
        self.dropout = nn.Dropout(config.resid_dropout)  # Dropout aplicado después de cada capa de encoder

    def forward(self, bytes_input, patch_boundaries=None):
        """
        Procesa la entrada de bytes a través del encoder local.

        Args:
            bytes_input (torch.Tensor): Entrada de bytes de forma [batch_size, seq_length].
            patch_boundaries (Optional[List[torch.Tensor]], optional): Lista de boundaries de parches. Por defecto es None.

        Returns:
            torch.Tensor: Salida del encoder local de forma [batch_size, seq_length, hidden_size].
        """
        print("\n[LocalEncoder] - Input shape:", bytes_input.shape)
        h = self.byte_embeddings(bytes_input)  # Generar embeddings para bytes
        print("[LocalEncoder] - After ByteEmbedding shape:", h.shape)
        h = self.embedding_dropout(h)  # Aplicar dropout a los embeddings
        
        # Generar tensor de posiciones para los tokens
        positions = torch.arange(bytes_input.size(1), device=bytes_input.device)
        
        # Pasar a través de cada capa del encoder local
        for idx, layer in enumerate(self.layers):
            print(f"[LocalEncoder] - Passing through EncoderLayer {idx}")
            h = layer(h, positions=positions)  # Aplicar capa de encoder
            print(f"[LocalEncoder] - EncoderLayer {idx} output shape:", h.shape)
            h = self.dropout(h)  # Aplicar dropout después de la capa de encoder
        print("[LocalEncoder] - Final output shape:", h.shape)
        return h  # Retornar la salida del encoder local


class GlobalTransformer(nn.Module):
    """
    Procesa la información a nivel de parches con atención global utilizando múltiples capas de encoder.
    Incorpora mecanismos avanzados de conexiones residuales con gates aprendibles y memoria jerárquica.

    Args:
        config (BLTConfig): Objeto de configuración que contiene los parámetros necesarios para el transformer global.

    Attributes:
        layers (nn.ModuleList): Lista de capas de encoder globales.
        dropout (nn.Dropout): Capa de dropout aplicada a las salidas de las capas.
        adaptive_dropout (nn.Dropout): Capa de dropout aplicada a los pesos adaptativos.
        gate_dropout (nn.Dropout): Capa de dropout aplicada a las gates.
        mem_dropout (nn.Dropout): Capa de dropout aplicada a la memoria jerárquica.
        skip_dropout (nn.Dropout): Capa de dropout aplicada a las conexiones de skip.
        adaptive_weights (nn.Parameter): Parámetros aprendibles para pesos adaptativos.
        layer_norms (nn.ModuleList): Lista de capas de normalización RMS para cada capa.
        skip_gates (nn.ModuleList): Lista de gates aprendibles para las conexiones de skip.
        hierarchical_mem (nn.Parameter): Memoria jerárquica para cada capa.
        mem_gate (nn.Sequential): Módulo de gate para la memoria jerárquica.
        input_norm (RMSNorm): Capa de normalización aplicada a la entrada.
        output_norm (RMSNorm): Capa de normalización aplicada a la salida.
    """
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.global_layers)])  # Capas de encoder globales

        self.dropout = nn.Dropout(config.resid_dropout)  # Dropout aplicado a las salidas de las capas
        self.adaptive_dropout = nn.Dropout(0.08)  # Dropout para los pesos adaptativos
        self.gate_dropout = nn.Dropout(0.08)  # Dropout para las gates
        self.mem_dropout = nn.Dropout(0.08)  # Dropout para la memoria jerárquica
        self.skip_dropout = nn.Dropout(0.08)  # Dropout para las conexiones de skip
        
        self.adaptive_weights = nn.Parameter(torch.ones(config.global_layers))  # Pesos adaptativos aprendibles
        self.layer_norms = nn.ModuleList([RMSNorm(config.hidden_size) for _ in range(config.global_layers)])  # Normalizaciones RMS por capa
        
        # Gates aprendibles para las conexiones de skip
        self.skip_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),  # Capa lineal para combinar entradas
                nn.Dropout(0.08),  # Dropout aplicado a la gate
                nn.Sigmoid()  # Activación sigmoid para gate
            ) for _ in range(config.global_layers)
        ])
        
        self.hierarchical_mem = nn.Parameter(torch.zeros(config.global_layers, config.hidden_size))  # Memoria jerárquica por capa
        self.mem_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),  # Capa lineal para combinar entradas y memoria
            nn.Dropout(0.08),  # Dropout aplicado al gate
            nn.Sigmoid()  # Activación sigmoid para gate
        )
        
        self.input_norm = RMSNorm(config.hidden_size)  # Normalización RMS aplicada a la entrada
        self.output_norm = RMSNorm(config.hidden_size)  # Normalización RMS aplicada a la salida

    def _apply_skip_connection(self, x, residual, layer_idx):
        """
        Aplica una conexión residual con gates aprendibles y pesos adaptativos.

        Args:
            x (torch.Tensor): Entrada actual de la capa.
            residual (torch.Tensor): Residual de la capa anterior.
            layer_idx (int): Índice de la capa actual.

        Returns:
            torch.Tensor: Salida después de aplicar la conexión residual.
        """
        print("[GlobalTransformer] - _apply_skip_connection")
        if x.dim() != residual.dim():
            residual = residual.view_as(x)  # Asegurar que residual tenga la misma forma que x
            
        norm_x = self.layer_norms[layer_idx](x)  # Normalizar la entrada actual
        norm_residual = self.layer_norms[layer_idx](residual)  # Normalizar el residual
        
        gate_input = torch.cat([norm_x, norm_residual], dim=-1)  # Concatenar para la gate
        gate = self.skip_gates[layer_idx](gate_input)  # Aplicar gate aprendido
        gate = self.gate_dropout(gate)  # Aplicar dropout al gate
        
        adaptive_weight = self.adaptive_dropout(self.adaptive_weights[layer_idx])  # Aplicar dropout al peso adaptativo
        adaptive_weight = adaptive_weight.view(1, 1, 1)  # Reorganizar dimensiones
        
        weighted_residual = adaptive_weight * gate * norm_residual  # Calcular residual ponderado
        weighted_residual = self.skip_dropout(weighted_residual)  # Aplicar dropout al residual ponderado
        
        out = x + weighted_residual  # Conexión residual
        print("[GlobalTransformer] - skip_connection output shape:", out.shape)
        return out  # Retornar la salida con la conexión residual aplicada

    def _apply_hierarchical_memory(self, x, layer_idx, batch_size):
        """
        Aplica la memoria jerárquica a la salida de la capa actual.

        Args:
            x (torch.Tensor): Salida actual de la capa.
            layer_idx (int): Índice de la capa actual.
            batch_size (int): Tamaño del batch.

        Returns:
            torch.Tensor: Salida después de aplicar la memoria jerárquica.
        """
        print("[GlobalTransformer] - _apply_hierarchical_memory")
        mem = self.hierarchical_mem[layer_idx:layer_idx+1]  # Obtener la memoria para la capa actual
        mem = mem.unsqueeze(0).expand(batch_size, x.size(1), -1)  # Expandir para el batch
        mem = self.mem_dropout(mem)  # Aplicar dropout a la memoria
        
        mem_gate_input = torch.cat([x, mem], dim=-1)  # Concatenar entrada y memoria para la gate
        mem_gate = self.mem_gate(mem_gate_input)  # Aplicar gate aprendido
        
        out = x + self.mem_dropout(mem_gate * mem)  # Combinar la memoria con la salida
        print("[GlobalTransformer] - hierarchical_memory output shape:", out.shape)
        return out  # Retornar la salida con memoria jerárquica aplicada

    def forward(self, patch_embeddings, attention_mask=None):
        """
        Procesa las representaciones de parches a través del transformer global.

        Args:
            patch_embeddings (torch.Tensor): Embeddings de parches de forma [batch_size, max_patches, hidden_size].
            attention_mask (Optional[torch.Tensor], optional): Máscara de atención de forma [batch_size, max_patches]. Por defecto es None.

        Returns:
            torch.Tensor: Salida del transformer global de forma [batch_size, max_patches, hidden_size].
        """
        print("\n[GlobalTransformer] - Input patch_embeddings shape:", patch_embeddings.shape)
        batch_size = patch_embeddings.size(0)  # Tamaño del batch
        h = self.dropout(self.input_norm(patch_embeddings))  # Aplicar normalización y dropout a la entrada
        print("[GlobalTransformer] - After input_norm shape:", h.shape)
        
        positions = torch.arange(patch_embeddings.size(1), device=patch_embeddings.device)  # Generar posiciones para los parches

        # Pasar a través de cada capa del transformer global
        for idx, layer in enumerate(self.layers):
            print(f"[GlobalTransformer] - Passing through EncoderLayer {idx}")
            prev_h = self.dropout(h)  # Guardar estado previo para conexiones residuales
            h = layer(h, self_mask=attention_mask, positions=positions)  # Aplicar capa de encoder
            h = self._apply_hierarchical_memory(h, idx, batch_size)  # Aplicar memoria jerárquica
            h = self._apply_skip_connection(h, prev_h, idx)  # Aplicar conexión residual
            h = self.dropout(h)  # Aplicar dropout después de la capa
            print(f"[GlobalTransformer] - EncoderLayer {idx} output shape:", h.shape)
        
        h = self.dropout(self.output_norm(h))  # Aplicar normalización y dropout a la salida global
        print("[GlobalTransformer] - Final output shape:", h.shape)
        return h  # Retornar la salida del transformer global

# =============================================================================
#                    MODELO DE ENTROPÍA (SIN USO DE CACHÉ)
# =============================================================================

class EntropyLM(nn.Module):
    """
    Modelo de entropía liviano optimizado con skip connections y gates aprendibles,
    SIN uso de cache. Calcula la entropía de las distribuciones de probabilidad
    para determinar dinámicamente los boundaries de parcheo.

    Args:
        hidden_size (int, opcional): Dimensionalidad oculta del modelo. Por defecto es 512.
        num_layers (int, opcional): Número de capas en el encoder. Por defecto es 2.
        num_heads (int, opcional): Número de cabezas de atención. Por defecto es 8.
        context_size (int, opcional): Tamaño del contexto para la atención. Por defecto es 512.
        dropout (float, opcional): Tasa de dropout. Por defecto es 0.1.
        learnable_dropout (float, opcional): Tasa de dropout para los parámetros aprendibles. Por defecto es 0.15.
        max_seq_length (int, opcional): Longitud máxima de la secuencia. Por defecto es 4096.
        window_size (int, opcional): Tamaño de la ventana para el cálculo de entropía. Por defecto es 128.

    Attributes:
        context_size (int): Tamaño del contexto para la atención.
        hidden_size (int): Dimensionalidad oculta del modelo.
        window_size (int): Tamaño de la ventana para el cálculo de entropía.
        byte_embedding (nn.Embedding): Capa de embedding para bytes individuales.
        encoder (nn.TransformerEncoder): Componente principal del encoder basado en Transformer.
        skip_weights (nn.ParameterList): Lista de pesos aprendibles para conexiones de skip.
        skip_dropout (nn.Dropout): Capa de dropout aplicada a las conexiones de skip.
        gate_layers (nn.ModuleList): Lista de capas de gates adaptativos para las conexiones de skip.
        output_gate (nn.Sequential): Módulo de gate para la conexión residual final.
        output (nn.Linear): Capa lineal para generar logits de bytes.
        dropout (nn.Dropout): Capa de dropout aplicada en el forward.
        learnable_dropout (nn.Dropout): Capa de dropout aplicada a los parámetros aprendibles.
        layer_norm (nn.LayerNorm): Capa de normalización aplicada a la entrada.
        cached_masks (torch.Tensor): Máscaras causales pre-generadas para diferentes longitudes de secuencia.
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

        self.context_size = context_size  # Tamaño del contexto para la atención
        self.hidden_size = hidden_size  # Dimensionalidad oculta del modelo
        self.window_size = window_size  # Tamaño de la ventana para el cálculo de entropía
        
        # Embedding inicial para bytes
        self.byte_embedding = nn.Embedding(256, hidden_size)  # Embeddings para bytes individuales
        
        # Definir una capa de encoder Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,  # Usar batch_first=True para compatibilidad con la entrada
            norm_first=True  # Normalizar antes de la atención y feedforward
        )

        # Crear el encoder Transformer con múltiples capas
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size),
            enable_nested_tensor=True  # Habilitar tensors anidados para eficiencia
        )
        
        # Crear una lista de parámetros aprendibles para las conexiones de skip
        self.skip_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, hidden_size))
            for _ in range(num_layers)
        ])
        self.skip_dropout = nn.Dropout(learnable_dropout)  # Dropout para las conexiones de skip
        
        # Crear gates adaptativos para las conexiones de skip
        self.gate_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),  # Combinar la entrada y el residual
                nn.Dropout(learnable_dropout),  # Aplicar dropout a la gate
                nn.Sigmoid()  # Activación sigmoid para gate
            )
            for _ in range(num_layers)
        ])
        
        # Memoria jerárquica para cada capa
        self.hierarchical_mem = nn.Parameter(
            torch.zeros(num_layers, hidden_size)
        )
        self.mem_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # Combinar la salida y la memoria
            nn.Dropout(learnable_dropout),  # Aplicar dropout a la gate
            nn.Sigmoid()  # Activación sigmoid para gate
        )
        
        # Capa de normalización aplicada a la entrada
        self.input_norm = RMSNorm(hidden_size)
        # Capa de normalización aplicada a la salida
        self.output_norm = RMSNorm(hidden_size)

    def _initialize_mask_cache(self, max_seq_length: int):
        """
        Inicializa y almacena máscaras causales para diferentes longitudes de secuencia.

        Args:
            max_seq_length (int): Longitud máxima de la secuencia.
        """
        masks = torch.triu(
            torch.ones(max_seq_length, max_seq_length),
            diagonal=1
        ).bool()  # Crear máscara triangular superior para causalidad
        self.register_buffer('cached_masks', masks, persistent=False)  # Almacenar como buffer

    def _apply_skip_connection(self, x: torch.Tensor, skip: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Aplica una conexión residual con gates aprendibles.

        Args:
            x (torch.Tensor): Entrada actual de la capa.
            skip (torch.Tensor): Residual de la capa anterior.
            layer_idx (int): Índice de la capa actual.

        Returns:
            torch.Tensor: Salida después de aplicar la conexión residual.
        """
        weighted_skip = self.skip_dropout(self.skip_weights[layer_idx]) * skip  # Ponderar el residual con weights aprendibles y aplicar dropout
        gate_input = torch.cat([x, weighted_skip], dim=-1)  # Concatenar la entrada y el residual ponderado
        gate = self.gate_layers[layer_idx](gate_input)  # Calcular gate aprendible
        out = x + self.learnable_dropout(gate * weighted_skip)  # Aplicar gate y conexión residual con dropout
        return out  # Retornar la salida con la conexión residual aplicada

    def _apply_hierarchical_memory(self, x, layer_idx, batch_size):
        """
        Aplica la memoria jerárquica a la salida de la capa actual.

        Args:
            x (torch.Tensor): Salida actual de la capa.
            layer_idx (int): Índice de la capa actual.
            batch_size (int): Tamaño del batch.

        Returns:
            torch.Tensor: Salida después de aplicar la memoria jerárquica.
        """
        print("[GlobalTransformer] - _apply_hierarchical_memory")
        mem = self.hierarchical_mem[layer_idx:layer_idx+1]  # Obtener la memoria para la capa actual
        mem = mem.unsqueeze(0).expand(batch_size, x.size(1), -1)  # Expandir para el batch
        mem = self.mem_dropout(mem)  # Aplicar dropout a la memoria
        
        mem_gate_input = torch.cat([x, mem], dim=-1)  # Concatenar la salida y la memoria para la gate
        mem_gate = self.mem_gate(mem_gate_input)  # Calcular gate aprendible
        
        out = x + self.mem_dropout(mem_gate * mem)  # Combinar la memoria con la salida con dropout
        print("[GlobalTransformer] - hierarchical_memory output shape:", out.shape)
        return out  # Retornar la salida con memoria jerárquica aplicada

    def forward(self, patch_embeddings, attention_mask=None):
        """
        Procesa las representaciones de parches a través del transformer global.

        Args:
            patch_embeddings (torch.Tensor): Embeddings de parches de forma [batch_size, max_patches, hidden_size].
            attention_mask (Optional[torch.Tensor], optional): Máscara de atención de forma [batch_size, max_patches]. Por defecto es None.

        Returns:
            torch.Tensor: Salida del transformer global de forma [batch_size, max_patches, hidden_size].
        """
        print("\n[GlobalTransformer] - Input patch_embeddings shape:", patch_embeddings.shape)
        batch_size = patch_embeddings.size(0)  # Tamaño del batch
        h = self.dropout(self.input_norm(patch_embeddings))  # Aplicar normalización y dropout a la entrada
        print("[GlobalTransformer] - After input_norm shape:", h.shape)
        
        positions = torch.arange(
            patch_embeddings.size(1), 
            device=patch_embeddings.device
        )  # Generar posiciones para los parches

        # Pasar a través de cada capa del transformer global
        for idx, layer in enumerate(self.layers):
            print(f"[GlobalTransformer] - Passing through EncoderLayer {idx}")
            prev_h = self.dropout(h)  # Guardar estado previo para conexiones residuales
            h = layer(h, self_mask=attention_mask, positions=positions)  # Aplicar capa de encoder
            h = self._apply_hierarchical_memory(h, idx, batch_size)  # Aplicar memoria jerárquica
            h = self._apply_skip_connection(h, prev_h, idx)  # Aplicar conexión residual
            h = self.dropout(h)  # Aplicar dropout después de la capa
            print(f"[GlobalTransformer] - EncoderLayer {idx} output shape:", h.shape)
        
        h = self.dropout(self.output_norm(h))  # Aplicar normalización y dropout a la salida global
        print("[GlobalTransformer] - Final output shape:", h.shape)
        return h  # Retornar la salida del transformer global


class LocalDecoder(nn.Module):
    """
    Decoder local que reconvierte las representaciones latentes en logits de bytes.
    Utiliza múltiples capas de decoder y una capa lineal final para predecir los bytes.

    Args:
        config (BLTConfig): Objeto de configuración que contiene los parámetros necesarios para el decoder local.

    Attributes:
        layers (nn.ModuleList): Lista de capas de decoder locales.
        byte_predictor (nn.Linear): Capa lineal para predecir los logits de bytes.
        dropout (nn.Dropout): Capa de dropout aplicada después de cada capa de decoder.
    """
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])  # Capas de decoder locales
        self.byte_predictor = nn.Linear(config.hidden_size, 256)  # Capa para predecir logits de bytes
        self.dropout = nn.Dropout(config.resid_dropout)  # Dropout aplicado después de cada capa de decoder

    def forward(self, encoded_bytes, global_output, byte_mask=None, cross_mask=None):
        """
        Reconstruye los logits de bytes a partir de las representaciones codificadas y la salida global.

        Args:
            encoded_bytes (torch.Tensor): Representaciones codificadas de bytes de forma [batch_size, seq_length, hidden_size].
            global_output (torch.Tensor): Salida global de forma [batch_size, max_patches, hidden_size].
            byte_mask (Optional[torch.Tensor], optional): Máscara para self-attention en los bytes. Por defecto es None.
            cross_mask (Optional[torch.Tensor], optional): Máscara para cross-attention. Por defecto es None.

        Returns:
            torch.Tensor: Logits de bytes de forma [batch_size, seq_length, 256].
        """
        print("\n[LocalDecoder] - Input encoded_bytes shape:", encoded_bytes.shape)
        h = encoded_bytes  # Inicializar la representación
        positions = torch.arange(encoded_bytes.size(1), device=encoded_bytes.device)  # Generar posiciones para los bytes
        
        # Pasar a través de cada capa del decoder local
        for idx, layer in enumerate(self.layers):
            print(f"[LocalDecoder] - Passing through DecoderLayer {idx}")
            h = layer(h, global_output, self_mask=byte_mask, cross_mask=cross_mask, positions=positions)  # Aplicar capa de decoder
            print(f"[LocalDecoder] - DecoderLayer {idx} output shape:", h.shape)
            h = self.dropout(h)  # Aplicar dropout después de la capa de decoder
        logits = self.byte_predictor(h)  # Predecir logits de bytes
        print("[LocalDecoder] - Final logits shape:", logits.shape)
        return logits  # Retornar los logits de bytes


# =============================================================================
#                           BLT (BYTE-LEVEL TRANSFORMER)
# =============================================================================

class BLT(nn.Module):
    """
    Byte-Level Transformer (BLT) con parcheo adaptativo optimizado,
    SIN usar cache en ningún lugar. Este modelo procesa texto a nivel de bytes
    utilizando un sistema de parcheo que aprende automáticamente los umbrales óptimos
    durante el entrenamiento. Incorpora un encoder local, un transformer global y
    un decoder local, junto con un modelo de entropía para determinar boundaries de parches
    de manera adaptativa.

    El modelo incluye parámetros aprendibles para el sistema de parcheo, permitiendo que se
    adapte automáticamente a diferentes tipos de contenido y patrones en los datos.

    Args:
        config (BLTConfig): Objeto de configuración que contiene todos los parámetros necesarios para el modelo.
                             Debe incluir parámetros para el tamaño del modelo, configuración de capas,
                             y parámetros específicos para el parcheo adaptativo.

    Attributes:
        config (BLTConfig): Configuración del modelo BLT.
        local_encoder (LocalEncoder): Encoder local que procesa los bytes detalladamente.
        global_transformer (GlobalTransformer): Transformer global que procesa la información a nivel de parches.
        local_decoder (LocalDecoder): Decoder local que reconvierte las representaciones latentes en logits de bytes.
        entropy_model (EntropyLM): Modelo de entropía para determinar boundaries de parches de manera adaptativa.
        global_norm (RMSNorm): Capa de normalización aplicada a la salida global.
        global_dropout (nn.Dropout): Capa de dropout aplicada a la salida global.
        min_patch_size (int): Tamaño mínimo permitido para un parche.
        max_patch_size (int): Tamaño máximo permitido para un parche.
        learnable_base_threshold (nn.Parameter): Umbral base aprendible para determinar boundaries.
        learnable_std_scale (nn.Parameter): Factor de escala aprendible para la desviación estándar.
        window_size (int): Tamaño de la ventana para cálculos estadísticos.
        stats_buffer (dict): Buffer para almacenar estadísticas internas (actualmente no utilizado).
        param_dropout (nn.Dropout): Capa de dropout aplicada a los parámetros aprendibles.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config  # Guardar configuración del modelo
        
        # Componentes principales del modelo
        self.local_encoder = LocalEncoder(config)  # Encoder local para procesar bytes detalladamente
        self.global_transformer = GlobalTransformer(config)  # Transformer global para atención a nivel de parches
        self.local_decoder = LocalDecoder(config)  # Decoder local para predecir bytes
        
        # Modelo de entropía para determinar boundaries de parches de manera adaptativa
        self.entropy_model = EntropyLM(
            hidden_size=config.hidden_size,
            num_layers=config.entropy_model_layers,
            num_heads=config.num_heads,
            context_size=config.entropy_context_size,
            dropout=config.attention_dropout
        )
        
        # Capas de normalización y dropout para la salida global
        self.global_norm = RMSNorm(config.hidden_size)  # Normalización RMS aplicada a la salida global
        self.global_dropout = nn.Dropout(config.resid_dropout)  # Dropout aplicado a la salida global
        
        # Parámetros base para el control de parches
        self.min_patch_size = config.min_patch_size  # Tamaño mínimo de un parche
        self.max_patch_size = config.max_patch_size  # Tamaño máximo de un parche
        
        # Parámetros aprendibles para el parcheo adaptativo
        self.learnable_base_threshold = nn.Parameter(
            torch.tensor(config.initial_entropy_threshold)
        )  # Umbral base aprendible
        self.learnable_std_scale = nn.Parameter(
            torch.tensor(0.5)
        )  # Factor de escala aprendible para la desviación estándar
        
        self.window_size = 128  # Tamaño de la ventana para cálculos estadísticos
        self.stats_buffer = {}  # Buffer para estadísticas internas (actualmente no utilizado)
        
        self.param_dropout = nn.Dropout(p=0.1)  # Dropout para parámetros aprendibles

    def _compute_adaptive_threshold(self, entropies: torch.Tensor) -> torch.Tensor:
        """
        Calcula el umbral adaptativo para determinar boundaries basados en la entropía.

        Args:
            entropies (torch.Tensor): Tensor de entropías de forma [batch_size, seq_length].

        Returns:
            torch.Tensor: Umbral adaptativo de forma [batch_size, 1].
        """
        print("[BLT] - _compute_adaptive_threshold")
        batch_size = entropies.size(0)  # Tamaño del batch
        means = []  # Lista para almacenar medias de ventanas
        stds = []  # Lista para almacenar desviaciones estándar de ventanas
        
        # Calcular medias y desviaciones estándar para cada ejemplo en el batch
        for i in range(batch_size):
            windows = entropies[i].unfold(0, self.window_size, self.window_size // 2)  # Dividir en ventanas deslizantes
            window_mean = windows.mean(dim=1, keepdim=True)  # Calcular media por ventana
            window_std = torch.sqrt(
                torch.var(windows, dim=1, keepdim=True, unbiased=False) + 1e-6  # Calcular varianza y convertir a desviación estándar
            )
            means.append(window_mean.mean())  # Agregar media de todas las ventanas
            stds.append(window_std.mean())  # Agregar desviación estándar de todas las ventanas
        
        mean = torch.stack(means).view(batch_size, 1)  # Stack de medias
        std = torch.stack(stds).view(batch_size, 1)  # Stack de desviaciones estándar
        
        # Aplicar dropout a los parámetros aprendibles
        base_threshold = self.param_dropout(self.learnable_base_threshold)  # Umbral base con dropout
        std_scale = self.param_dropout(self.learnable_std_scale)  # Escala de desviación estándar con dropout
        
        # Calcular el umbral adaptativo usando una función sigmoide para mantener valores entre 0 y 1
        threshold = torch.sigmoid(
            base_threshold + std_scale * std
        ).clamp(min=0.1, max=0.9)  # Limitar el umbral entre 0.1 y 0.9
        
        print("[BLT] - Adaptive threshold shape:", threshold.shape)
        return threshold  # Retornar el umbral adaptativo

    def compute_patches(self, bytes_input: torch.Tensor) -> List[torch.Tensor]:
        """
        Calcula los boundaries de los parches usando un sistema híbrido que combina parcheo base
        y adaptativo aprendible. El sistema inicia con un parcheo base simple durante las etapas
        tempranas del entrenamiento y transiciona gradualmente a un parcheo basado en entropía
        más sofisticado conforme el modelo mejora.

        El parcheo base utiliza un tamaño fijo determinado por min_patch_size, lo que proporciona
        estabilidad inicial. El parcheo adaptativo usa el modelo de entropía para determinar 
        boundaries basados en el contenido, permitiendo una segmentación más inteligente.

        Args:
            bytes_input (torch.Tensor): Tensor de entrada de bytes con forma [batch_size, seq_length].
                                        Contiene los bytes a ser procesados y parcheados.

        Returns:
            List[torch.Tensor]: Lista de tensores de boundaries para cada ejemplo en el batch.
                                Cada tensor contiene las posiciones donde se dividen los parches.

        Notas Técnicas:
            - En etapas tempranas (< 20% del entrenamiento): Usa parcheo base con stride fijo
            - En etapas posteriores: Usa modelo de entropía con umbral adaptativo
            - El dropout (p=0.1) se aplica solo durante entrenamiento
            - Los boundaries siempre respetan min_patch_size y max_patch_size
            - Las operaciones de entropía se realizan con precisión mixta (amp)
            
        Ejemplo de uso:
            model = BLT(config)
            input_tensor = torch.tensor([...])  # [batch_size, seq_length]
            boundaries = model.compute_patches(input_tensor)
            # boundaries[0] contiene los límites para el primer ejemplo del batch

        Comportamiento en diferentes fases:
            1. Fase Inicial (< 20% entrenamiento):
               - Usa divisiones fijas basadas en min_patch_size
               - Más estable y predecible
               - Menor costo computacional

            2. Fase Adaptativa (> 20% entrenamiento):
               - Usa modelo de entropía para boundaries
               - Se adapta al contenido
               - Permite patches de tamaño variable

            3. Inferencia:
               - Siempre usa el modelo de entropía
               - Máxima adaptabilidad al contenido
        """
        print("[BLT] - compute_patches")
        batch_size = bytes_input.size(0)  # Obtener el tamaño del batch
        boundaries_list = []  # Lista para almacenar boundaries de cada ejemplo
        
        # Calcular progreso del entrenamiento (0 al inicio, 1 al final o en inferencia)
        training_progress = getattr(self, 'current_step', 0) / getattr(self, 'total_steps', 1000) \
                            if self.training else 1.0

        # Fase inicial: Usar parcheo base simple
        if training_progress < 0.2:
            for i in range(batch_size):
                # Crear boundaries con stride fijo basado en min_patch_size
                base_boundaries = torch.arange(
                    self.min_patch_size, 
                    bytes_input.size(1), 
                    self.min_patch_size, 
                    device=bytes_input.device
                )
                boundaries_list.append(base_boundaries)  # Agregar boundaries al batch
            return boundaries_list  # Retornar lista de boundaries

        # Fase adaptativa: Usar modelo de entropía
        # Habilitar autocast para operaciones de entropía (mejora eficiencia y estabilidad)
        with torch.cuda.amp.autocast():
            # Calcular entropías para toda la secuencia
            entropies = self.entropy_model(bytes_input)
            # Aplicar dropout para regularización (solo en entrenamiento)
            entropies = F.dropout(entropies, p=0.1, training=self.training)

        # Obtener umbrales adaptativos para cada ejemplo en el batch
        thresholds = self._compute_adaptive_threshold(entropies)

        # Procesar cada ejemplo en el batch
        for i in range(batch_size):
            current_entropies = entropies[i]  # Entropías del ejemplo actual
            threshold = thresholds[i]  # Umbral adaptativo para el ejemplo actual

            boundaries = []  # Lista para almacenar boundaries del ejemplo
            current_size = 0  # Tamaño actual del parche
            last_boundary = 0  # Última posición de boundary

            # Analizar cada posición para determinar boundaries
            for pos, entropy in enumerate(current_entropies):
                current_size = pos - last_boundary + 1  # Actualizar tamaño del parche

                # Determinar si crear un nuevo parche basado en:
                # 1. Si excede el tamaño máximo permitido
                # 2. Si supera el tamaño mínimo y la entropía supera el umbral
                should_split = any([
                    current_size >= self.max_patch_size,
                    current_size >= self.min_patch_size and entropy > threshold
                ])

                if should_split:
                    boundaries.append(pos)  # Agregar posición como boundary
                    last_boundary = pos + 1  # Actualizar última posición de boundary

            # Convertir lista de boundaries a tensor y mover al dispositivo correcto
            boundaries_list.append(
                torch.tensor(boundaries, device=bytes_input.device)
            )

        return boundaries_list  # Retornar lista de boundaries

    def forward(self, bytes_input: torch.Tensor, patch_boundaries: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Realiza el paso hacia adelante del modelo BLT con parcheo adaptativo aprendible.

        Args:
            bytes_input (torch.Tensor): Entrada de bytes de forma [batch_size, seq_length].
            patch_boundaries (Optional[List[torch.Tensor]], optional): Lista de boundaries precomputados. Por defecto es None.

        Returns:
            torch.Tensor: Logits de salida de forma [batch_size, seq_length, 256].
        """
        print("\n[BLT] - Forward start")
        batch_size, seq_length = bytes_input.size()  # Obtener dimensiones del batch
        
        byte_encodings = self.local_encoder(bytes_input)  # Procesar bytes con encoder local
        byte_encodings = self.global_dropout(byte_encodings)  # Aplicar dropout a las representaciones codificadas
        
        if patch_boundaries is None:
            patch_boundaries = self.compute_patches(bytes_input)  # Calcular boundaries si no se proporcionan
        
        patch_means = []  # Lista para almacenar medias de patches
        for i in range(batch_size):
            boundaries = patch_boundaries[i]  # Obtener boundaries para el ejemplo actual
            patches = []  # Lista para almacenar patches del ejemplo
            start = 0  # Posición inicial del parche

            if boundaries.numel() > 0:
                # Asegurarse de que boundaries es 1-D
                boundaries = boundaries if boundaries.dim() > 0 else boundaries.unsqueeze(0)
                # Añadir la posición final de la secuencia como último boundary
                boundaries = torch.cat([
                    boundaries,
                    torch.tensor([seq_length], device=boundaries.device)
                ])
            else:
                # Si no hay boundaries, considerar toda la secuencia como un solo parche
                boundaries = torch.tensor([seq_length], device=bytes_input.device)

            # Dividir la secuencia en patches basados en boundaries
            for end in boundaries:
                if end > start:
                    patch = byte_encodings[i, start:end].mean(dim=0)  # Calcular la media de cada patch
                    patches.append(patch)  # Agregar patch a la lista
                    start = end  # Actualizar posición inicial para el siguiente patch

            if not patches:
                patches.append(byte_encodings[i].mean(dim=0))  # Asegurar al menos un patch

            patch_means.append(torch.stack(patches))  # Agregar patches del ejemplo al batch

        # Encontrar el número máximo de patches en el batch para padding
        max_patches = max(p.size(0) for p in patch_means)
        # Crear un tensor de zeros para almacenar los patches con padding
        padded_patches = byte_encodings.new_zeros(
            batch_size, max_patches, self.config.hidden_size
        )
        
        # Rellenar padded_patches con los patches reales
        for i, patches in enumerate(patch_means):
            num_patches = patches.size(0)  # Número de patches en el ejemplo
            padded_patches[i, :num_patches] = patches  # Asignar patches al tensor de salida

        # Procesar los patches con el transformer global
        global_output = self.global_transformer(padded_patches)  # Salida del transformer global
        global_output = self.global_norm(global_output)  # Aplicar normalización RMS
        global_output = self.global_dropout(global_output)  # Aplicar dropout
        
        # Decodificar las representaciones para predecir logits de bytes
        logits = self.local_decoder(
            self.global_dropout(byte_encodings),  # Aplicar dropout a las representaciones codificadas
            global_output  # Pasar la salida global al decoder
        )

        print("[BLT] - Forward end, logits shape:", logits.shape)
        return logits  # Retornar los logits de salida

# =============================================================================
#                       CONFIGURACIÓN DEL MODELO BLT
# =============================================================================

class BLTConfig:
    """
    Configuración del Byte-Level Transformer (BLT).
    Permite ajustar fácilmente los hiperparámetros del modelo sin modificar directamente el código de las clases.

    Args:
        hidden_size (int, opcional): Tamaño de la capa oculta. Por defecto es 256.
        intermediate_size (int, opcional): Tamaño de la capa intermedia en las capas feedforward. Por defecto es 1024.
        num_heads (int, opcional): Número de cabezas de atención. Por defecto es 4.
        encoder_layers (int, opcional): Número de capas en el encoder local. Por defecto es 1.
        global_layers (int, opcional): Número de capas en el transformer global. Por defecto es 6.
        decoder_layers (int, opcional): Número de capas en el decoder local. Por defecto es 2.
        attention_dropout (float, opcional): Tasa de dropout para la atención. Por defecto es 0.12.
        resid_dropout (float, opcional): Tasa de dropout para las conexiones residuales. Por defecto es 0.12.
        ngram_vocab_size (int, opcional): Tamaño del vocabulario para los n-gramas. Por defecto es 10000.
        window_size (int, opcional): Tamaño de la ventana para cálculos internos. Por defecto es 512.
        max_position_embeddings (int, opcional): Número máximo de embeddings posicionales. Por defecto es 1024.
        entropy_model_layers (int, opcional): Número de capas en el modelo de entropía. Por defecto es 2.
        entropy_context_size (int, opcional): Tamaño del contexto para el modelo de entropía. Por defecto es 512.
        entropy_threshold (float, opcional): Umbral de entropía para el parcheo adaptativo. Por defecto es 0.5.
        min_patch_size (int, opcional): Tamaño mínimo de un parche. Por defecto es 32.
        max_patch_size (int, opcional): Tamaño máximo de un parche. Por defecto es 512.
        initial_entropy_threshold (float, opcional): Umbral inicial de entropía. Por defecto es 0.5.

    Attributes:
        hidden_size (int): Tamaño de la capa oculta.
        intermediate_size (int): Tamaño de la capa intermedia en las capas feedforward.
        num_heads (int): Número de cabezas de atención.
        encoder_layers (int): Número de capas en el encoder local.
        global_layers (int): Número de capas en el transformer global.
        decoder_layers (int): Número de capas en el decoder local.
        attention_dropout (float): Tasa de dropout para la atención.
        resid_dropout (float): Tasa de dropout para las conexiones residuales.
        ngram_vocab_size (int): Tamaño del vocabulario para los n-gramas.
        window_size (int): Tamaño de la ventana para cálculos internos.
        max_position_embeddings (int): Número máximo de embeddings posicionales.
        entropy_model_layers (int): Número de capas en el modelo de entropía.
        entropy_context_size (int): Tamaño del contexto para el modelo de entropía.
        entropy_threshold (float): Umbral de entropía para el parcheo adaptativo.
        min_patch_size (int): Tamaño mínimo de un parche.
        max_patch_size (int): Tamaño máximo de un parche.
        initial_entropy_threshold (float): Umbral inicial de entropía.
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
        self.hidden_size = hidden_size  # Tamaño de la capa oculta
        self.intermediate_size = intermediate_size  # Tamaño de la capa intermedia
        self.num_heads = num_heads  # Número de cabezas de atención
        self.encoder_layers = encoder_layers  # Número de capas en el encoder local
        self.global_layers = global_layers  # Número de capas en el transformer global
        self.decoder_layers = decoder_layers  # Número de capas en el decoder local
        self.attention_dropout = attention_dropout  # Dropout para la atención
        self.resid_dropout = resid_dropout  # Dropout para conexiones residuales
        self.ngram_vocab_size = ngram_vocab_size  # Tamaño del vocabulario para n-gramas
        self.window_size = window_size  # Tamaño de la ventana para cálculos internos
        self.max_position_embeddings = max_position_embeddings  # Número máximo de embeddings posicionales
        self.entropy_model_layers = entropy_model_layers  # Número de capas en el modelo de entropía
        self.entropy_context_size = entropy_context_size  # Tamaño del contexto para el modelo de entropía
        self.entropy_threshold = entropy_threshold  # Umbral de entropía para parcheo adaptativo
        self.min_patch_size = min_patch_size  # Tamaño mínimo de un parche
        self.max_patch_size = max_patch_size  # Tamaño máximo de un parche
        self.initial_entropy_threshold = initial_entropy_threshold  # Umbral inicial de entropía


# =============================================================================
#                         UTILIDADES DE MÁSCARAS
# =============================================================================

def create_block_causal_mask(seq_length: int, window_size: int = None):
    """
    Crea una máscara causal con ventana opcional (sin cache).
    La máscara permite que cada posición atienda solo a posiciones anteriores
    y, opcionalmente, limita la atención a un rango específico (window_size).

    Args:
        seq_length (int): Longitud de la secuencia.
        window_size (int, opcional): Tamaño de la ventana para limitar la atención. Por defecto es None.

    Returns:
        torch.Tensor: Máscara causal de forma [seq_length, seq_length].
    """
    mask = torch.ones((seq_length, seq_length), dtype=torch.bool)  # Crear una matriz de unos
    mask = torch.triu(mask, diagonal=1)  # Mantener solo la parte triangular superior para causalidad
    
    if window_size:
        indices = torch.arange(seq_length)  # Crear un tensor de índices
        window_mask = (indices.unsqueeze(1) - indices.unsqueeze(0)).abs() <= window_size  # Crear máscara de ventana
        mask = mask | ~window_mask  # Combinar máscaras para limitar el rango de atención
    
    return mask  # Retornar la máscara causal


def create_patch_mask(patch_boundaries, seq_length):
    """
    Crea una máscara para parches con optimizaciones de memoria y soporte
    para parches dinámicos (sin cache).
    La máscara identifica bloques de parches donde la atención está permitida
    dentro de cada bloque.

    Args:
        patch_boundaries (Union[List[int], torch.Tensor]): Lista o tensor de boundaries de parches.
        seq_length (int): Longitud total de la secuencia.

    Returns:
        torch.Tensor: Máscara de parches de forma [seq_length, seq_length].
    """
    if isinstance(patch_boundaries, list):
        patch_boundaries = torch.tensor(patch_boundaries)  # Convertir lista a tensor
    elif patch_boundaries.numel() == 0:
        return torch.zeros((seq_length, seq_length), dtype=torch.bool)  # Retornar máscara de ceros si no hay boundaries
    
    if not torch.all(patch_boundaries[1:] > patch_boundaries[:-1]):
        patch_boundaries, _ = torch.sort(patch_boundaries)  # Ordenar boundaries si no están ordenados
    
    mask = torch.zeros((seq_length, seq_length), dtype=torch.bool)  # Inicializar máscara de ceros
    
    start = 0  # Posición inicial
    for end in patch_boundaries:
        if end > start:
            mask[start:end, start:end] = True  # Permitir atención dentro del parche
            start = end  # Actualizar posición inicial para el siguiente parche
    
    if start < seq_length:
        mask[start:, start:] = True  # Permitir atención en el último parche si es necesario
    
    return mask  # Retornar la máscara de parches


# =============================================================================
#          CONFIGURACIÓN PARA PARCHEO (PatchingConfig) Y FUNCIONES AUX
# =============================================================================

class PatchingConfig:
    """
    Configuración para el esquema de parcheo.

    Permite definir diferentes esquemas de parcheo como 'entropy', 'space', o 'fixed',
    ajustando los parámetros necesarios para cada uno.

    Args:
        scheme (str, opcional): Esquema de parcheo a utilizar ('entropy', 'space', 'fixed'). Por defecto es 'entropy'.
        entropy_threshold (float, opcional): Umbral de entropía para el parcheo adaptativo. Por defecto es 0.5.
        stride (int, opcional): Tamaño del stride para el parcheo fijo. Por defecto es 4.
        reset_context (bool, opcional): Indica si se debe resetear el contexto después de cada parche. Por defecto es True.
        use_monotonic (bool, opcional): Indica si se debe usar un parcheo monotónico en el esquema de entropía. Por defecto es True.

    Attributes:
        scheme (str): Esquema de parcheo a utilizar.
        entropy_threshold (float): Umbral de entropía para el parcheo adaptativo.
        stride (int): Tamaño del stride para el parcheo fijo.
        reset_context (bool): Indica si se debe resetear el contexto después de cada parche.
        use_monotonic (bool): Indica si se debe usar un parcheo monotónico en el esquema de entropía.
    """
    def __init__(
        self,
        scheme='entropy',
        entropy_threshold=0.5,
        stride=4,
        reset_context=True,
        use_monotonic=True
    ):
        self.scheme = scheme  # Esquema de parcheo a utilizar
        self.entropy_threshold = entropy_threshold  # Umbral de entropía para el parcheo adaptativo
        self.stride = stride  # Tamaño del stride para el parcheo fijo
        self.reset_context = reset_context  # Indica si se debe resetear el contexto después de cada parche
        self.use_monotonic = use_monotonic  # Indica si se debe usar un parcheo monotónico en el esquema de entropía


# =============================================================================
#                         FUNCIONES DE ENTRENAMIENTO
# =============================================================================

def train_step(model, optimizer, batch, patch_config):
    """
    Realiza un paso de entrenamiento, calculando la pérdida y aplicando backpropagation.

    Args:
        model (BLT): Modelo BLT a entrenar.
        optimizer (torch.optim.Optimizer): Optimizador para actualizar los parámetros del modelo.
        batch (torch.Tensor): Batch de datos de entrada de forma [batch_size, seq_length].
        patch_config (PatchingConfig): Configuración para el parcheo a utilizar durante el entrenamiento.

    Returns:
        float: Valor de la pérdida calculada para el batch.
    """
    optimizer.zero_grad()  # Resetear gradientes del optimizador

    input_bytes = batch[:, :-1]  # Entradas de bytes (todos excepto el último)
    target_bytes = batch[:, 1:]  # Targets de bytes (todos excepto el primero)
    patch_boundaries = None  # Inicializar boundaries

    # Seleccionar esquema de parcheo basado en patch_config
    if patch_config.scheme == 'entropy':
        with torch.no_grad():  # No calcular gradientes para entropía
            entropies = model.entropy_model(input_bytes)  # Calcular entropías
            indices = torch.where(entropies > patch_config.entropy_threshold)  # Encontrar posiciones con entropía alta
            if indices[0].numel() == 0:
                patch_boundaries = torch.tensor([], dtype=torch.long, device=entropies.device)  # No hay boundaries
            else:
                patch_boundaries = indices[1]  # Posiciones de boundaries

    elif patch_config.scheme == 'space':
        indices = torch.where(input_bytes == 32)  # Encontrar espacios (byte 32)
        if indices[0].numel() == 0:
            patch_boundaries = torch.tensor([], dtype=torch.long, device=input_bytes.device)  # No hay boundaries
        else:
            patch_boundaries = indices[1] + 1  # Ajustar boundaries después del espacio

    elif patch_config.scheme == 'fixed':
        stride = patch_config.stride  # Obtener stride fijo
        seq_length = input_bytes.size(1)  # Longitud de la secuencia
        patch_boundaries = torch.arange(
            stride, seq_length, stride, device=input_bytes.device
        )  # Crear boundaries con stride fijo

    logits = model(input_bytes, patch_boundaries)  # Obtener logits del modelo

    # Reshape para calcular la pérdida
    logits_reshaped = logits.view(-1, logits.size(-1))  # [batch_size * seq_length, vocab_size]
    target_bytes_reshaped = target_bytes.view(-1)  # [batch_size * seq_length]
    loss = F.cross_entropy(logits_reshaped, target_bytes_reshaped)  # Calcular pérdida de entropía cruzada

    loss.backward()  # Backpropagation
    optimizer.step()  # Actualizar parámetros del modelo

    return loss.item()  # Retornar el valor de la pérdida


# =============================================================================
#                           FUNCIÓN DE GENERACIÓN
# =============================================================================

def generate(model, start_bytes, max_length=1000, temperature=1.0, top_k=20, patch_config=None, device='cpu'):
    """
    Genera una secuencia de bytes a partir de un contexto inicial.

    Utiliza técnicas de muestreo controladas por 'temperature' y 'top_k' para diversificar las salidas.
    Aplica parcheo adaptativo basado en la configuración proporcionada.

    Args:
        model (BLT): Modelo BLT para la generación.
        start_bytes (List[int] o torch.Tensor): Lista o tensor de bytes iniciales.
        max_length (int, opcional): Longitud máxima de la secuencia a generar. Por defecto es 1000.
        temperature (float, opcional): Temperatura para el muestreo de probabilidades. Por defecto es 1.0.
        top_k (int, opcional): Número de tokens más probables a considerar durante el muestreo. Por defecto es 20.
        patch_config (Optional[PatchingConfig], opcional): Configuración para el parcheo durante la generación. Por defecto es None.
        device (str, opcional): Dispositivo de PyTorch donde se realizará la generación. Por defecto es 'cpu'.

    Returns:
        List[int]: Lista de bytes generados.
    """
    model.eval()  # Establecer el modelo en modo evaluación
    generated = list(start_bytes)  # Inicializar la lista de bytes generados con los bytes iniciales
    
    with torch.no_grad():  # Desactivar cálculo de gradientes
        while len(generated) < max_length:  # Continuar hasta alcanzar max_length
            input_bytes = torch.tensor(generated, device=device).unsqueeze(0)  # Crear tensor de entrada [1, len]
            
            # Determinar boundaries de parcheo basados en patch_config
            if patch_config is not None:
                if patch_config.scheme == 'entropy':
                    entropies = model.entropy_model(input_bytes)  # Calcular entropías
                    if patch_config.use_monotonic:
                        entropy_diff = entropies[:, 1:] - entropies[:, :-1]  # Diferencia de entropías
                        patch_boundaries = torch.where(entropy_diff > patch_config.entropy_threshold)[1] + 1  # Encontrar boundaries
                    else:
                        patch_boundaries = torch.where(entropies > patch_config.entropy_threshold)[1]  # Encontrar boundaries
                else:
                    patch_boundaries = torch.arange(
                        patch_config.stride, 
                        len(generated), 
                        patch_config.stride,
                        device=device
                    )  # Crear boundaries con stride fijo
            else:
                patch_boundaries = None  # No usar boundaries si no se proporciona patch_config
            
            logits = model(input_bytes, patch_boundaries)  # Obtener logits del modelo

            # Ajustar la forma de logits para muestreo
            if logits.dim() == 3:
                logits = logits[0, -1] / temperature  # [vocab_size]
            elif logits.dim() == 2:
                logits = logits[0] / temperature  # [vocab_size]
            else:
                break  # Salir si la forma no es esperada
            
            # Aplicar top-k muestreo
            top_k = min(top_k, logits.size(-1))  # Limitar top_k al tamaño del vocabulario
            topk_logits, topk_indices = torch.topk(logits, top_k)  # Obtener top_k logits e índices
            topk_probs = F.softmax(topk_logits, dim=-1)  # Calcular probabilidades para top_k
            
            # Seleccionar el siguiente byte basado en las probabilidades
            next_byte = topk_indices[torch.multinomial(topk_probs, 1).item()].item()
            generated.append(next_byte)  # Agregar el byte generado a la lista
            
            if next_byte == 0:  # Si se genera un byte nulo (EOF)
                break  # Terminar la generación
                
    return generated  # Retornar la lista de bytes generados


# =============================================================================
#           CÁLCULO DE BITS POR BYTE (PERPLEJIDAD / ENTROPÍA)
# =============================================================================

def compute_bits_per_byte(model, data, patch_config):
    """
    Calcula bits por byte para un conjunto de datos dado, útil como métrica.
    Esta métrica mide la entropía promedio por byte, lo que indica la eficiencia del modelo
    en la representación de la información.

    Args:
        model (BLT): Modelo BLT entrenado.
        data (Iterable[torch.Tensor]): Conjunto de datos de entrada, cada batch de forma [batch_size, seq_length].
        patch_config (PatchingConfig): Configuración para el parcheo a utilizar durante la evaluación.

    Returns:
        float: Bits por byte calculados para el conjunto de datos.
    """
    model.eval()  # Establecer el modelo en modo evaluación
    total_loss = 0  # Inicializar pérdida total
    total_bytes = 0  # Inicializar contador de bytes
    
    with torch.no_grad():  # Desactivar cálculo de gradientes
        for batch in data:
            input_bytes = batch[:, :-1]  # Entradas de bytes
            target_bytes = batch[:, 1:]  # Targets de bytes
            
            # Seleccionar esquema de parcheo basado en patch_config
            if patch_config.scheme == 'entropy':
                entropies = model.entropy_model(input_bytes.unsqueeze(0))  # Calcular entropías
                patch_boundaries = torch.where(entropies > patch_config.entropy_threshold)[2]  # Encontrar boundaries
            else:
                patch_boundaries = torch.arange(patch_config.stride, input_bytes.size(0), patch_config.stride)  # Crear boundaries con stride fijo
            
            logits = model(input_bytes.unsqueeze(0), [patch_boundaries])  # Obtener logits del modelo
            loss = F.cross_entropy(logits.reshape(-1, 256), target_bytes.reshape(-1), reduction='sum')  # Calcular pérdida de entropía cruzada con suma
            
            total_loss += loss.item()  # Acumular la pérdida
            total_bytes += target_bytes.numel()  # Acumular el número de bytes
    
    bits_per_byte = total_loss / (total_bytes * math.log(2))  # Convertir la pérdida a bits por byte
    return bits_per_byte  # Retornar la métrica
