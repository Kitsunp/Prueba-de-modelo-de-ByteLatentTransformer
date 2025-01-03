import os
import torch

# Habilitar Flash SDP explícitamente
torch.backends.cuda.enable_flash_sdp(enabled=True)
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# Verificar si está habilitado
print(f"Flash SDP habilitado: {torch.backends.cuda.flash_sdp_enabled()}")

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from blt_model import BLT, BLTConfig, PatchingConfig, train_step, generate
import json
from pathlib import Path
import logging
from typing import Dict, Any, Tuple, List
import math
class ByteDataset(Dataset):
    def __init__(self, 
                 texts: List[str], 
                 max_length: int = 1024, 
                 min_length: int = 30,
                 report_stats: bool = True) -> None:
        """
        Dataset para procesar textos a nivel de bytes con filtrado y limpieza integrados.
        
        Args:
            texts: Lista de textos a procesar
            max_length: Longitud máxima permitida en bytes para cada texto
            min_length: Longitud mínima permitida en caracteres para cada texto
            report_stats: Si se deben mostrar estadísticas del proceso de limpieza
        """
        self.max_length = max_length
        self.min_length = min_length
        self.encoded_texts = []
        self.stats = {}
        
        # Estadísticas iniciales
        self.stats['initial_count'] = len(texts)
        self._process_texts(texts, report_stats)
        
    def _count_empty_lines(self, texts: List[str]) -> Tuple[int, int]:
        """Cuenta las líneas vacías en los textos."""
        empty_count = sum(1 for text in texts if len(text.strip()) == 0)
        return empty_count, len(texts)
    
    def _clean_empty_texts(self, texts: List[str]) -> List[str]:
        """Elimina los textos vacíos o que solo contienen espacios."""
        return [text for text in texts if text.strip() != ""]
    
    def _filter_short_texts(self, texts: List[str]) -> List[str]:
        """Filtra los textos que son más cortos que min_length."""
        return [text for text in texts if len(text) >= self.min_length]
    
    def _process_texts(self, texts: List[str], report_stats: bool) -> None:
        """Procesa los textos aplicando filtros y limpieza."""
        # Contar líneas vacías iniciales
        empty_count, total_count = self._count_empty_lines(texts)
        self.stats['initial_empty'] = empty_count
        self.stats['initial_total'] = total_count
        
        if report_stats:
            print(f"\nEstadísticas de procesamiento de textos:")
            print(f"Textos iniciales: {total_count}")
            print(f"Líneas vacías iniciales: {empty_count} ({(empty_count/total_count)*100:.2f}%)")
        
        # Limpiar textos vacíos
        texts = self._clean_empty_texts(texts)
        self.stats['after_cleaning'] = len(texts)
        
        if report_stats:
            print(f"Textos después de limpieza: {len(texts)}")
        
        # Filtrar textos cortos
        texts = self._filter_short_texts(texts)
        self.stats['after_length_filter'] = len(texts)
        
        if report_stats:
            print(f"Textos después de filtrado por longitud mínima: {len(texts)}")
        
        # Codificar textos
        skipped_count = 0
        for text in tqdm(texts, desc="Codificando textos a bytes"):
            try:
                bytes_encoding = text.encode('utf-8')
                if len(bytes_encoding) > self.max_length:
                    bytes_encoding = bytes_encoding[:self.max_length]
                bytes_array = np.frombuffer(bytes_encoding, dtype=np.uint8).copy()
                self.encoded_texts.append(bytes_array)
            except Exception as e:
                skipped_count += 1
                continue
        
        self.stats['encoding_failed'] = skipped_count
        self.stats['final_count'] = len(self.encoded_texts)
        
        if report_stats:
            print(f"Textos que fallaron en codificación: {skipped_count}")
            print(f"Textos codificados finales: {len(self.encoded_texts)}")
            print(f"Reducción total: {((1 - len(self.encoded_texts)/total_count) * 100):.2f}%\n")
    
    def get_stats(self) -> Dict:
        """Retorna las estadísticas del procesamiento."""
        return self.stats
    
    def __len__(self) -> int:
        return len(self.encoded_texts)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.encoded_texts[idx], dtype=torch.long)

def collate_batch(examples, max_length=None):
    # Manejar caso de lista vacía
    if not examples:
        return torch.tensor([])
    
    # Si max_length es None, encontrar la longitud máxima
    if max_length is not None:
        examples = [ex[:max_length] for ex in examples]
        max_len = max_length
    else:
        # Cambio clave: usar max_length si no se especifica
        max_len = max(len(ex) for ex in examples)
    
    padded = []
    for ex in examples:
        if len(ex) < max_len:
            # Cambio clave: usar dtype del tensor original
            padding = torch.zeros(max_len - len(ex), dtype=ex.dtype)
            ex = torch.cat([ex, padding])
        ex = ex.contiguous()
        padded.append(ex)
    
    return torch.stack(padded).contiguous()
from colorama import Fore, Back, Style, init
from datetime import datetime
import time
from tqdm import tqdm
import os

# Inicializar colorama
init()

def format_time(seconds):
    """Formatea segundos en una cadena legible."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def print_header(text):
    """Imprime un encabezado formateado."""
    width = os.get_terminal_size().columns
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*width}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{text.center(width)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*width}{Style.RESET_ALL}\n")

def format_value(value, format_spec=None):
    """Formatea un valor numérico."""
    try:
        if format_spec:
            if isinstance(value, (int, float)):
                return f"{value:{format_spec}}"
        return str(value)
    except:
        return str(value)
def limit_logits_confidence(logits, max_conf=0.94):
    """
    Limita confianza directamente en logits
    Método más profundo y efectivo
    """
    # Calcular valor máximo de logits correspondiente a max_conf
    max_log_prob = torch.log(torch.tensor(max_conf))
    
    # Normalizar logits
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Limitar logits
    limited_log_probs = torch.where(
        log_probs > max_log_prob, 
        max_log_prob, 
        log_probs
    )
    
    # Renormalizar para mantener distribución
    limited_log_probs = limited_log_probs - limited_log_probs.logsumexp(dim=-1, keepdim=True)
    
    return limited_log_probs
def print_metrics_table(metrics, title=None):
    """Imprime una tabla de métricas."""
    if title:
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}{title}{Style.RESET_ALL}")
    width = max(len(name) for name, _ in metrics) + 15
    print(f"{Fore.BLUE}{Style.BRIGHT}{'─'*width}{Style.RESET_ALL}")
    for name, (value, format_spec, color) in metrics:
        value_str = format_value(value, format_spec)
        print(f"{color}{name.ljust(30)}{Style.BRIGHT}{value_str}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT}{'─'*width}{Style.RESET_ALL}")
import torch_optimizer as optim  # Importa la librería con AdaBelief

def train_model(model, train_loader, val_loader, config, num_epochs=10):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_header(f"Iniciando entrenamiento - {device}")
    
    scaler = torch.cuda.amp.GradScaler()
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.005)
    )
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.15)
    
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
    num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader) * 2, T_mult=2, eta_min=1e-6
    )
    
    patch_config = PatchingConfig(
        scheme='entropy',
        entropy_threshold=0.5,
        use_monotonic=True
    )
    
    best_val_loss = float('inf')
    patience = config.get('patience', 3)
    patience_counter = 0
    total_steps = 0
    
    def validate_model():
        model.eval()
        metrics = {'loss': 0, 'steps': 0, 'correct': 0, 'total': 0, 'confidences': []}
        
        progress_bar = tqdm(val_loader, desc=f'{Fore.BLUE}Validación{Style.RESET_ALL}',
                          leave=False, ncols=100)
        
        with torch.no_grad():
            for batch in progress_bar:
                try:
                    batch = batch.to(device, non_blocking=True)
                    input_bytes = batch[:, :-1]
                    target_bytes = batch[:, 1:]
                    
                    with torch.cuda.amp.autocast():
                        if patch_config.scheme == 'entropy':
                            entropies = model.entropy_model(input_bytes)
                            patch_boundaries = torch.where(entropies > patch_config.entropy_threshold)[0]
                        else:
                            patch_boundaries = None
                        
                        logits = model(input_bytes, patch_boundaries)
                        
                        loss = F.cross_entropy(logits.reshape(-1, 256), target_bytes.reshape(-1))
                        
                        probabilities = F.softmax(logits, dim=-1)
                        predictions = torch.argmax(probabilities, dim=-1)
                        correct = (predictions == target_bytes).float()
                        confidence = probabilities.gather(-1, target_bytes.unsqueeze(-1)).squeeze(-1)
                    
                    metrics['loss'] += loss.item()
                    metrics['steps'] += 1
                    metrics['correct'] += correct.sum().item()
                    metrics['total'] += correct.numel()
                    metrics['confidences'].extend(confidence.detach().cpu().numpy())
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        print(f'{Fore.RED}WARNING: Out of memory durante la validación. Saltando batch{Style.RESET_ALL}')
                        continue
                    raise e
                finally:
                    del_vars = ['loss', 'logits', 'patch_boundaries', 'entropies']
                    for var in del_vars:
                        if var in locals(): del locals()[var]
        
        results = {
            'loss': metrics['loss'] / metrics['steps'] if metrics['steps'] > 0 else float('inf'),
            'accuracy': metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0,
            'perplexity': np.exp(metrics['loss'] / metrics['steps']) if metrics['steps'] > 0 else float('inf'),
            'confidence_min': np.min(metrics['confidences']) if metrics['confidences'] else 0,
            'confidence_max': np.max(metrics['confidences']) if metrics['confidences'] else 0,
            'confidence_mean': np.mean(metrics['confidences']) if metrics['confidences'] else 0
        }
        
        return results
    
    print_header("Configuración del entrenamiento")
    config_metrics = [
        ("Épocas", (num_epochs, None, Fore.GREEN)),
        ("Learning Rate", (config['learning_rate'], ".2e", Fore.GREEN)),
        ("Batch Size", (config['batch_size'], None, Fore.GREEN)),
        ("Gradient Accumulation", (gradient_accumulation_steps, None, Fore.GREEN))
    ]
    print_metrics_table(config_metrics)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print_header(f"Época {epoch+1}/{num_epochs}")
        
        model.train()
        train_metrics = {'loss': 0, 'steps': 0, 'correct': 0, 'total': 0, 'confidences': []}
        optimizer.zero_grad(set_to_none=True)
        
        desc = f'{Fore.GREEN}Entrenamiento{Style.RESET_ALL}'
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=desc, leave=False, ncols=100)
        
        for step, batch in progress_bar:
            # Añadir aquí la actualización del progreso
            current_step = epoch * len(train_loader) + step
            model.update_training_progress(current_step, num_epochs * len(train_loader))
            # Calcular el factor de mezcla actual (similar al cálculo en compute_patches)
            training_progress = current_step / (num_epochs * len(train_loader))
            if training_progress < 0.2:
                entropy_factor = 0.0
            elif training_progress > 0.8:
                entropy_factor = 1.0
            else:
                entropy_factor = (training_progress - 0.2) / 0.6
                entropy_factor = 1 / (1 + math.exp(-10 * (entropy_factor - 0.5)))
                        
            if step % 100 == 0:
                torch.cuda.empty_cache()

            try:
                if batch.size(1) > config.get('max_sequence_length', 1024):
                    batch = batch[:, :config['max_sequence_length']]
                batch = batch.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    input_bytes = batch[:, :-1]
                    target_bytes = batch[:, 1:]
                    
                    patch_boundaries = None
                    if patch_config.scheme == 'entropy':
                        with torch.no_grad():
                            entropies = model.entropy_model(input_bytes)
                            patch_boundaries = torch.where(entropies > patch_config.entropy_threshold)[0]
                    
                    logits = model(input_bytes, patch_boundaries)
                    limited_logits = limit_logits_confidence(logits, max_conf=0.9)
                    loss = criterion(limited_logits.reshape(-1, 256), target_bytes.reshape(-1))
                    loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    if config.get('max_grad_norm'):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    total_steps += 1
                
                # Actualizar métricas
                train_metrics['loss'] += loss.item() * gradient_accumulation_steps
                train_metrics['steps'] += 1
                
                probabilities = F.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                correct = (predictions == target_bytes).float()
                confidence = probabilities.gather(-1, target_bytes.unsqueeze(-1)).squeeze(-1)             
                train_metrics['correct'] += correct.sum().item()
                train_metrics['total'] += correct.numel()
                train_metrics['confidences'].extend(confidence.detach().cpu().numpy())
                
                # Actualizar barra de progreso
                current_loss = train_metrics['loss']/train_metrics['steps']
                current_lr = optimizer.param_groups[0]['lr']
                current_mem = torch.cuda.memory_allocated()/1024**3
                
                # Actualizar la barra de progreso con el factor de entropía
                progress_bar.set_postfix({
                    'loss': f"{current_loss:.4f}",
                    'lr': f"{current_lr:.2e}",
                    #'mem': f"{current_mem:.1f}GB",
                    'EntropyLM': f"{entropy_factor*100:.1f}%"  # Nuevo campo
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    print(f'{Fore.RED}WARNING: Out of memory en batch {step}. Saltando batch{Style.RESET_ALL}')
                    continue
                raise e
            finally:
                del_vars = ['loss', 'logits', 'patch_boundaries', 'entropies']
                for var in del_vars:
                    if var in locals(): del locals()[var]
        
        # Calcular métricas de entrenamiento
        train_results = {
            'loss': train_metrics['loss'] / train_metrics['steps'],
            'accuracy': train_metrics['correct'] / train_metrics['total'],
            'perplexity': np.exp(train_metrics['loss'] / train_metrics['steps']),
            'confidence_min': np.min(train_metrics['confidences']),
            'confidence_max': np.max(train_metrics['confidences']),
            'confidence_mean': np.mean(train_metrics['confidences'])
        }
        
        # Validación
        val_results = validate_model()
        
        # Mostrar resultados
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        
        info_metrics = [
            ("Tiempo de época", (format_time(epoch_time), None, Fore.YELLOW)),
            ("Tiempo total", (format_time(total_time), None, Fore.YELLOW)),
            ("Learning Rate", (optimizer.param_groups[0]['lr'], ".2e", Fore.YELLOW)),
            ("Memoria GPU (GB)", (torch.cuda.memory_allocated()/1024**3, ".1f", Fore.YELLOW))
        ]
        print_metrics_table(info_metrics, "Información General")
        
        train_metrics_table = [
            ("Pérdida", (train_results['loss'], ".4f", Fore.GREEN)),
            ("Accuracy", (train_results['accuracy'], ".4f", Fore.GREEN)),
            ("Perplejidad", (train_results['perplexity'], ".4f", Fore.GREEN)),
            ("Confianza Media", (train_results['confidence_mean'], ".4f", Fore.GREEN)),
            ("Confianza Mín", (train_results['confidence_min'], ".4f", Fore.GREEN)),
            ("Confianza Máx", (train_results['confidence_max'], ".4f", Fore.GREEN))
        ]
        print_metrics_table(train_metrics_table, "Métricas de Entrenamiento")
        
        val_metrics_table = [
            ("Pérdida", (val_results['loss'], ".4f", Fore.BLUE)),
            ("Accuracy", (val_results['accuracy'], ".4f", Fore.BLUE)),
            ("Perplejidad", (val_results['perplexity'], ".4f", Fore.BLUE)),
            ("Confianza Media", (val_results['confidence_mean'], ".4f", Fore.BLUE)),
            ("Confianza Mín", (val_results['confidence_min'], ".4f", Fore.BLUE)),
            ("Confianza Máx", (val_results['confidence_max'], ".4f", Fore.BLUE))
        ]
        print_metrics_table(val_metrics_table, "Métricas de Validación")
        
        # Guardar mejor modelo
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'val_results': val_results,
                'config': config,
                'total_steps': total_steps
            }
            
            try:
                torch.save(checkpoint, 'best_blt_model.pt')
                print(f"\n{Fore.GREEN}✓ Guardado nuevo mejor modelo con pérdida de validación: {val_results['loss']:.4f}{Style.RESET_ALL}")
            except Exception as e:
                print(f"\n{Fore.RED}✗ No se pudo guardar el checkpoint: {str(e)}{Style.RESET_ALL}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n{Fore.YELLOW}! Early stopping activado después de {patience} épocas sin mejora{Style.RESET_ALL}")
                break
        
        torch.cuda.empty_cache()
    
    return model, best_val_loss
def load_model(model_path, config):
    """
    Carga un modelo BLT desde un checkpoint, manejando diferentes formatos de state_dict.
    
    Args:
        model_path (str): Ruta al archivo del checkpoint
        config (BLTConfig): Configuración del modelo
        
    Returns:
        tuple: (modelo cargado, checkpoint)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model = BLT(config)
        
        # Obtener el state dict del checkpoint
        state_dict = checkpoint['model_state_dict']
        
        # Verificar si las claves tienen el prefijo '_orig_mod'
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            # Crear un nuevo state dict sin el prefijo
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        # Filtrar claves no existentes y cargar el estado
        model_keys = set(model.state_dict().keys())
        state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        
        # Cargar los pesos filtrados
        missing_keys = model.load_state_dict(state_dict, strict=False)
        
        # Imprimir información sobre claves faltantes
        if missing_keys.missing_keys:
            print("\nAdvertencia: Algunas claves no se encontraron en el checkpoint:")
            print(f"Claves faltantes: {len(missing_keys.missing_keys)}")
        
        # Mover el modelo al dispositivo correcto
        model = model.to(device)
        model.eval()
        
        return model, checkpoint
        
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        raise e
def prepare_input_bytes(text, device):
    bytes_encoding = text.encode('utf-8')
    bytes_array = np.frombuffer(bytes_encoding, dtype=np.uint8)
    return torch.tensor(bytes_array, dtype=torch.long).unsqueeze(0).to(device)
def generate_text(model, start_text, max_length=1000, temperature=1.0, patch_config=None, top_k=50):
    """
    Genera texto utilizando el modelo BLT con técnicas avanzadas de muestreo y beam search.
    
    Args:
        model: El modelo BLT entrenado.
        start_text (str): El texto de inicio para la generación.
        max_length (int): Longitud máxima de la respuesta en bytes.
        temperature (float): Factor de temperatura para el muestreo (0.0-1.0).
        patch_config (PatchingConfig): Configuración de parcheo para el modelo.
        top_k (int): Número de tokens top-k para considerar en el muestreo.
    
    Returns:
        str: El texto generado.
    
    Features:
        - Beam Search con adaptación dinámica
        - Nucleus Sampling (top-p) con temperatura adaptativa
        - Caché de estados para generación eficiente
        - Manejo robusto de errores y excepciones
        - Control de calidad de salida
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from dataclasses import dataclass
    from typing import List, Optional, Tuple
    import math
    
    @dataclass
    class GenerationCache:
        """Cache para almacenar estados intermedios durante la generación."""
        hidden_states: Optional[torch.Tensor] = None
        attention_mask: Optional[torch.Tensor] = None
        position_ids: Optional[torch.Tensor] = None

    class BeamHypothesis:
        """Clase para manejar hipótesis en beam search."""
        def __init__(self, sequence: List[int], score: float):
            self.sequence = sequence
            self.score = score

        def __lt__(self, other):
            return self.score < other.score

    def compute_adaptive_temperature(logits: torch.Tensor) -> float:
        """Calcula temperatura adaptativa basada en la entropía de los logits."""
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return min(1.0, max(0.3, entropy.item() / 4.0))

    def nucleus_sampling(logits: torch.Tensor, temperature: float, top_p: float = 0.9) -> int:
        """Realiza nucleus sampling (top-p) con temperatura."""
        # [DEBUG] Iniciar nucleus_sampling
        #print("[DEBUG] Iniciando nucleus_sampling.")
        scaled_logits = logits / temperature
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Crear máscara para top-p
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Aplicar máscara y renormalizar
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        probs = F.softmax(sorted_logits, dim=-1)

        # Muestrear y asegurar que retornamos un int
        item = torch.multinomial(probs, 1)
        selected_index = sorted_indices[item].item()
        
        # [DEBUG] Token seleccionado
        #print(f"[DEBUG] Token seleccionado: {selected_index}")
        
        return int(selected_index)  # Convertir explícitamente a int

    def beam_search_step(
        model,
        input_ids: torch.Tensor,
        beam_scores: torch.Tensor,
        cache: GenerationCache,
        beam_width: int,
        patch_boundaries_arg,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor, GenerationCache]:
        """Realiza un paso de beam search."""
        # [DEBUG] Iniciar beam_search_step
        #print("[DEBUG] Iniciando beam_search_step.")
        logits = model(input_ids, patch_boundaries_arg)
        #print(f"[DEBUG] Logits shape after model forward: {logits.shape}")
        
        next_token_logits = logits[:, -1, :] / temperature
        #print(f"[DEBUG] Next token logits shape: {next_token_logits.shape}")
        
        # Calcular puntuaciones para todos los posibles siguientes tokens
        vocab_size = next_token_logits.size(-1)
        next_scores = F.log_softmax(next_token_logits, dim=-1) + beam_scores.unsqueeze(-1)
        next_scores = next_scores.view(-1)
        #print(f"[DEBUG] Next scores shape after reshaping: {next_scores.shape}")
        
        # Asegurar que beam_width no exceda el tamaño del vocabulario
        effective_beam_width = min(beam_width, next_scores.size(0))
        #print(f"[DEBUG] Effective beam width: {effective_beam_width}")
        
        # Seleccionar los mejores beam_width tokens
        top_scores, top_tokens = torch.topk(next_scores, effective_beam_width)
        #print(f"[DEBUG] Top scores: {top_scores}")
        #print(f"[DEBUG] Top tokens: {top_tokens}")
        
        # Calcular índices evitando la división problemática
        beam_indices = torch.div(top_tokens, vocab_size, rounding_mode='floor')
        next_tokens = top_tokens % vocab_size
        #print(f"[DEBUG] Beam indices: {beam_indices}")
        #print(f"[DEBUG] Next tokens: {next_tokens}")
        
        # Actualizar secuencias
        current_sequences = input_ids[beam_indices]
        next_sequences = torch.cat([current_sequences, next_tokens.unsqueeze(1)], dim=-1)
        #print(f"[DEBUG] Next sequences shape: {next_sequences.shape}")
        
        return next_sequences, top_scores, cache

    # Inicio de la lógica principal de generate_text
    model.eval()
    device = next(model.parameters()).device
    #print("[DEBUG] Modelo en modo evaluación.")
    #print(f"[DEBUG] Dispositivo del modelo: {device}")

    try:
        # Convertir entrada a bytes y verificar
        input_bytes = start_text.encode('utf-8')
        input_ids = list(input_bytes)
        #print(f"[DEBUG] Input bytes: {input_bytes}")
        #print(f"[DEBUG] Input IDs: {input_ids}")
        
        if not input_ids:
            #print("[DEBUG] Entrada vacía detectada.")
            return "Error: entrada vacía"
        
        # Configuración de generación
        beam_width = 5
        top_p = 0.9
        min_length = 30
        repetition_penalty = 1.2
        cache = GenerationCache()
        #print("[DEBUG] Configuración de generación inicializada.")
        
        with torch.no_grad():
            # Inicialización para beam search
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            beam_scores = torch.zeros(1, device=device)
            sequences = [(input_ids, 0.0)]
            #print(f"[DEBUG] Inicialización de input_tensor: {input_tensor.shape}")
            #print(f"[DEBUG] Beam scores iniciales: {beam_scores}")
            #print(f"[DEBUG] Secuencias iniciales: {sequences}")
            
            while len(sequences[0][0]) < max_length:
                # [DEBUG] Inicio del ciclo de generación
                #print(f"[DEBUG] Iteración de generación: {len(sequences[0][0])} < {max_length}")
                
                current_sequences = [seq for seq, _ in sequences]
                current_scores = torch.tensor([score for _, score in sequences], device=device)
                #print(f"[DEBUG] Current scores shape: {current_scores.shape}")
                #print(f"[DEBUG] Current scores: {current_scores}")
                
                # Preparar entrada para el modelo
                batch_sequences = torch.tensor(current_sequences, device=device)
                #print(f"[DEBUG] Batch sequences shape: {batch_sequences.shape}")
                
                # Calcular patch boundaries
                patch_boundaries = []
                if patch_config is not None and patch_config.scheme == 'entropy':
                    try:
                        #print("[DEBUG] Calculando entropías para patch boundaries.")
                        entropies = model.entropy_model(batch_sequences)
                        #print(f"[DEBUG] Entropies shape: {entropies.shape}")

                        if patch_config.use_monotonic and entropies.size(1) > 1:
                            entropy_diff = entropies[:, 1:] - entropies[:, :-1]
                            #print(f"[DEBUG] Entropy differences shape: {entropy_diff.shape}")
                            indices = torch.where(entropy_diff > patch_config.entropy_threshold)[1]
                            #print(f"[DEBUG] Entropy threshold indices: {indices}")
                            patch_boundaries = indices + 1 if indices.numel() > 0 else []
                        else:
                            indices = torch.where(entropies > patch_config.entropy_threshold)[1]
                            #print(f"[DEBUG] Entropy threshold indices (no monotonic): {indices}")
                            patch_boundaries = indices if indices.numel() > 0 else []

                        #print(f"[DEBUG] Patch boundaries para este batch: {patch_boundaries}")
                    except Exception as e:
                        #print(f"[DEBUG] Advertencia en cálculo de entropía: {e}")
                        patch_boundaries = []

                # Convertir patch_boundaries a argumento manejable
                if isinstance(patch_boundaries, torch.Tensor):
                    patch_boundaries_arg = patch_boundaries if patch_boundaries.numel() > 0 else None
                elif isinstance(patch_boundaries, list):
                    patch_boundaries_arg = patch_boundaries if len(patch_boundaries) > 0 else None
                else:
                    patch_boundaries_arg = None

                #print(f"[DEBUG] Patch boundaries argument: {patch_boundaries_arg}")
                
                # Alternar entre beam search y nucleus sampling
                if len(current_sequences[0]) < min_length:
                    # [DEBUG] Usando beam search para el inicio
                    #print("[DEBUG] Usando beam search para la generación.")
                    next_sequences, next_scores, cache = beam_search_step(
                        model, batch_sequences, current_scores,
                        cache, beam_width, patch_boundaries_arg, temperature
                    )
                    
                    # Actualizar secuencias
                    sequences = [
                        (seq.tolist(), score.item())
                        for seq, score in zip(next_sequences, next_scores)
                    ]
                    #print(f"[DEBUG] Nuevas secuencias después de beam search: {sequences}")
                else:
                    # [DEBUG] Usando nucleus sampling para más variedad
                    #print("[DEBUG] Usando nucleus sampling para la generación.")
                    logits = model(batch_sequences, patch_boundaries_arg)
                    #print(f"[DEBUG] Logits shape después del modelo: {logits.shape}")
                    
                    next_token_logits = logits[0, -1, :]
                    #print(f"[DEBUG] Next token logits shape: {next_token_logits.shape}")
                    
                    # Aplicar temperatura adaptativa
                    adaptive_temp = compute_adaptive_temperature(next_token_logits)
                    actual_temp = temperature * adaptive_temp
                    #print(f"[DEBUG] Adaptive temperature: {adaptive_temp}, Actual temperature: {actual_temp}")
                    
                    # Aplicar penalización por repetición
                    for seq in current_sequences:
                        for prev_token in seq[-20:]:  # Últimos 20 tokens
                            if prev_token >= 0 and prev_token < next_token_logits.size(-1):
                                next_token_logits[prev_token] /= repetition_penalty
                                #print(f"[DEBUG] Aplicando penalización por repetición al token: {prev_token}")
                    
                    # Muestrear el siguiente token
                    next_token = nucleus_sampling(
                        next_token_logits,
                        temperature=actual_temp,
                        top_p=top_p
                    )
                    #print(f"[DEBUG] Next token muestreado: {next_token}")
                    
                    # Actualizar secuencias
                    best_sequence = sequences[0][0]
                    best_sequence.append(next_token)
                    sequences = [(best_sequence, 0.0)]
                    #print(f"[DEBUG] Nueva secuencia después de nucleus sampling: {sequences}")
                
                # Verificar condiciones de parada
                if sequences[0][0][-1] == 0:
                    #print("[DEBUG] Token de parada (0) encontrado. Finalizando generación.")
                    break
                if len(sequences[0][0]) >= max_length:
                    #print("[DEBUG] Longitud máxima alcanzada. Finalizando generación.")
                    break
            
            # Seleccionar la mejor secuencia
            best_sequence = sequences[0][0]
            #print(f"[DEBUG] Secuencia final generada: {best_sequence}")
            
            # Post-procesamiento y control de calidad
            generated_bytes = bytes([b for b in best_sequence if b != 0])
            generated_text = generated_bytes.decode('utf-8', errors='replace').strip()
            #print(f"[DEBUG] Texto generado después de decodificación: {generated_text}")
            
            # Verificar calidad mínima
            if len(generated_text) < 10 or generated_text.isspace():
                #print("[DEBUG] Texto generado de baja calidad detectado.")
                return "Error: generación de texto de baja calidad"
            
            print("[DEBUG] Generación completada exitosamente.")
            return generated_text
                
    except Exception as e:
        print(f"[DEBUG] Error en generación: {str(e)}")
        return f"Error en generación: {str(e)}"

def main():
    # Configurar precisión para mejor rendimiento
    torch.set_float32_matmul_precision('high')

    # Habilitar Flash SDP explícitamente
    torch.backends.cuda.enable_flash_sdp(enabled=True)

    # Verificar si está habilitado
    print(f"Flash SDP habilitado: {torch.backends.cuda.flash_sdp_enabled()}")


    # Configurar semillas para reproducibilidad
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print(f"Usando CPU")

    # Cargar datasets
    print("Cargando datasets...")
    try:
        # Dataset de entrenamiento: OpenWebText
        train_dataset_stream = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        train_size = 40000
        train_texts = []
        print("Cargando dataset de entrenamiento (OpenWebText)...")
        for i, example in enumerate(train_dataset_stream):
            if i >= train_size:
                break
            train_texts.append(example["text"])
        print(f"Tamaño de entrenamiento (OpenWebText): {len(train_texts)}")
        
        # Dataset de validación: Wikitext-103
        val_size = 60000
    
        val_dataset_stream = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
        val_texts = []
        print("Cargando dataset de validación (Wikitext-103)...")
        for i, example in enumerate(val_dataset_stream):
            if i >= val_size:
                break
            val_texts.append(example["text"])
        print(f"Tamaño de validación (Wikitext-103): {len(val_texts)}")
    
    except Exception as e:
        print(f"Error al cargar los datasets: {str(e)}")
        return

    # Configuración del modelo
    model_config = BLTConfig(
        hidden_size=512,
        intermediate_size=2048,
        num_heads=16,
        encoder_layers=2,
        global_layers=8,
        decoder_layers=6,
        attention_dropout=0.13,
        resid_dropout=0.12,
        ngram_vocab_size=150000,
        window_size=512,
        max_position_embeddings=4096,
        entropy_model_layers=2,
        entropy_context_size=512,
        entropy_threshold=0.5
    )
                             
    # Configuración del entrenamiento
    training_config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.015,
        'warmup_steps': 500,
        'max_grad_norm': 2,
        'gradient_accumulation_steps': 4,
        'max_sequence_length': 1532,
        'batch_size': 8,
        'eval_batch_size': 8,
        'patience': 5,
        'num_epochs': 5,
        'min_text_length': 30
    } 

    try:
        print("\nCreando y procesando datasets...")
        print("\nProcesando dataset de entrenamiento (OpenWebText):")
        train_dataset = ByteDataset(
            train_texts, 
            max_length=training_config['max_sequence_length'],
            min_length=training_config['min_text_length'],
            report_stats=True
        )
        
        print("\nProcesando dataset de validación (Wikitext-103):")
        val_dataset = ByteDataset(
            val_texts, 
            max_length=training_config['max_sequence_length'],
            min_length=training_config['min_text_length'],
            report_stats=True
        )

        # Estadísticas de procesamiento
        train_stats = train_dataset.get_stats()
        val_stats = val_dataset.get_stats()

        # DataLoader para entrenamiento
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config['batch_size'],
            shuffle=True,
            collate_fn=lambda x: collate_batch(x, max_length=training_config['max_sequence_length']),
            num_workers=4,  # Reducido para mejor estabilidad
            pin_memory=True,
            prefetch_factor=2,
            drop_last=True,
            persistent_workers=True
        )

        # DataLoader para validación
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config['eval_batch_size'],
            shuffle=False,
            collate_fn=lambda x: collate_batch(x, max_length=training_config['max_sequence_length']),
            num_workers=4,  # Reducido para mejor estabilidad
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

        print("\nInicializando modelo...")
        model = BLT(model_config)
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total de parámetros: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,}")

        # Definir la ruta del modelo
        model_path = 'best_blt_model.pt'
    
        if os.path.exists(model_path):
            print(f"\nSe encontró un modelo entrenado en '{model_path}'.")
            print("¿Qué deseas hacer?")
            print("1. Usar el modelo existente para generar respuestas.")
            print("2. Entrenar un nuevo modelo.")
            
            while True:
                try:
                    choice = input("Ingresa el número de tu elección (1 o 2): ").strip()
                    if choice in ['1', '2']:
                        break
                    print("Entrada inválida. Por favor, ingresa '1' o '2'.")
                except Exception:
                    print("Error de entrada. Por favor, intenta de nuevo.")
            
            if choice == '1':
                try:
                    print("\nCargando el modelo existente...")
                    model, checkpoint = load_model(model_path, model_config)
                    print("Modelo cargado correctamente.")
                    
                    # Configuración de parcheo
                    patch_config = PatchingConfig(
                        scheme='entropy',
                        entropy_threshold=0.5,
                        use_monotonic=True
                    )
                    
                    print("\nIniciando sesión interactiva. Escribe 'salir' para terminar.")
                    while True:
                        try:
                            user_input = input("\nTu pregunta: ").strip()
                            if user_input.lower() in ['salir', 'exit', 'quit']:
                                print("Terminando la interacción.")
                                break
                            if not user_input:
                                print("Por favor, ingresa una pregunta válida.")
                                continue
                                
                            response = generate_text(
                                model=model,
                                start_text=user_input,
                                max_length=500,
                                temperature=1.0,
                                patch_config=patch_config,
                                top_k=50
                            )
                            print(f"Respuesta: {response}")
                            
                        except KeyboardInterrupt:
                            print("\nInteracción interrumpida por el usuario.")
                            break
                        except Exception as e:
                            print(f"Error generando respuesta: {str(e)}")
                            print("Intenta con una pregunta diferente.")
                    
                    return
                
                except Exception as e:
                    print(f"Error cargando el modelo: {str(e)}")
                    print("Continuando con el entrenamiento de un nuevo modelo...")
            
        print("\nIniciando entrenamiento de un nuevo modelo...")
        
        # Intentar compilar el modelo
        try:
            print("Compilando modelo...")
            model = torch.compile(model)
            print("Modelo compilado exitosamente.")
        except Exception as e:
            print(f"Advertencia: No se pudo compilar el modelo: {str(e)}")
            print("Continuando sin compilación...")
        
        # Entrenamiento
        try:
            model, best_val_loss = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=training_config,
                num_epochs=training_config['num_epochs']
            )
            
            print(f"\nEntrenamiento completado. Mejor pérdida de validación: {best_val_loss:.4f}")
            
            # Guardar configuración
            final_config = {
                'model_config': model_config.__dict__,
                'training_config': training_config,
                'best_val_loss': best_val_loss,
                'train_size': len(train_texts),
                'val_size': len(val_texts),
                'device': str(device),
                'total_params': total_params,
                'trainable_params': trainable_params,
                'data_processing_stats': {
                    'train': train_stats,
                    'val': val_stats
                }
            }
            
            with open('training_config.json', 'w') as f:
                json.dump(final_config, f, indent=4)
            
            # Guardar modelo sin compilación
            model = model._orig_mod if hasattr(model, '_orig_mod') else model
            checkpoint = {
                'epoch': training_config['num_epochs'],
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'config': model_config.__dict__
            }
            torch.save(checkpoint, model_path)
            print(f"Modelo guardado en '{model_path}'.")
            
            # Iniciar sesión interactiva
            patch_config = PatchingConfig(
                scheme='entropy',
                entropy_threshold=0.5,
                use_monotonic=True
            )
            
            print("\nIniciando sesión interactiva. Escribe 'salir' para terminar.")
            while True:
                try:
                    user_input = input("\nTu pregunta: ").strip()
                    if user_input.lower() in ['salir', 'exit', 'quit']:
                        print("Terminando la interacción.")
                        break
                    if not user_input:
                        print("Por favor, ingresa una pregunta válida.")
                        continue
                        
                    response = generate_text(
                        model=model,
                        start_text=user_input,
                        max_length=500,
                        temperature=1.0,
                        patch_config=patch_config,
                        top_k=50
                    )
                    print(f"Respuesta: {response}")
                    
                except KeyboardInterrupt:
                    print("\nInteracción interrumpida por el usuario.")
                    break
                except Exception as e:
                    print(f"Error generando respuesta: {str(e)}")
                    print("Intenta con una pregunta diferente.")
                    
        except KeyboardInterrupt:
            print("\nEntrenamiento interrumpido por el usuario!")
            # Guardar checkpoint de emergencia
            model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': training_config,
                'data_processing_stats': {
                    'train': train_stats,
                    'val': val_stats
                },
                'interrupted': True
            }, 'emergency_checkpoint.pt')
            print("Checkpoint de emergencia guardado!")
            
        except Exception as e:
            print(f"\nError durante el entrenamiento: {str(e)}")
            raise e

    except Exception as e:
        print(f"Error en la configuración: {str(e)}")
        raise e

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nLimpieza completada.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error fatal: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
