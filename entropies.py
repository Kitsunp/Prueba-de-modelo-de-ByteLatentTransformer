#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Archivo: entropies_plus_generation.py

Objetivo:
    - Extender la evaluación del modelo BLT (que integra EntropyLM) con:
      1) Distribución de entropías y conteo de parches (igual que antes).
      2) Métricas más ricas: tamaños de parche, entropía promedio por parche.
      3) Algunos ejemplos concretos de segmentación (texto real).
      4) Prueba de generación para medir si el modelo, con su parcheo, 
         mantiene habilidad de generar texto coherente.

Uso:
    python entropies_plus_generation.py 
        --model_path best_blt_model.pt 
        --batch_size 32 
        --samples_to_inspect 3 
        --assess_generation True 
        --generation_samples 3 
        [Otros argumentos]
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm

# Importamos las definiciones necesarias del proyecto actual
from blt_model import BLT, BLTConfig, PatchingConfig, generate
from train import load_model, ByteDataset, collate_batch
from torch.utils.data import DataLoader

# ==========================================================================
#                      CONFIGURACIÓN DEL MODELO GLOBAL
# ==========================================================================
model_config = BLTConfig(
    hidden_size=512,
    intermediate_size=2048,
    num_heads=16,
    encoder_layers=2,
    global_layers=8,
    decoder_layers=6,
    attention_dropout=0.13,
    resid_dropout=0.12,
    ngram_vocab_size=400000,
    window_size=256,
    max_position_embeddings=4096,
    entropy_model_layers=2,
    entropy_context_size=256,
    entropy_threshold=0.5,
    min_patch_size=32,
    max_patch_size=256,
    initial_entropy_threshold=0.5
)

# ==========================================================================
#                          FUNCIONES AUXILIARES
# ==========================================================================
def get_patch_sizes(boundaries, seq_length):
    """
    Dada una lista/array de boundaries y la longitud total de la secuencia,
    retorna una lista con el tamaño de cada parche.
    """
    if not isinstance(boundaries, (list, np.ndarray, torch.Tensor)):
        return []

    # Asegurar que boundaries esté ordenado y que incluya el final.
    boundaries_sorted = sorted(boundaries)
    if len(boundaries_sorted) == 0 or boundaries_sorted[-1] != seq_length:
        boundaries_sorted.append(seq_length)

    patch_sizes = []
    start = 0
    for end in boundaries_sorted:
        if end > start:
            patch_sizes.append(end - start)
            start = end
    return patch_sizes


def analyze_text_segmentations(
    text_list,
    model: BLT,
    device: torch.device,
    max_seq_length: int,
    num_samples: int = 3
):
    """
    Toma una pequeña muestra de `num_samples` secuencias, muestra:
      - El texto original (truncado).
      - Los boundaries generados por `model.compute_patches(...)`.
      - Los tamaños de parche.
      - Entropía promedio por parche.
    """
    if not text_list:
        return

    sample_indices = np.random.choice(len(text_list), size=min(num_samples, len(text_list)), replace=False)
    print("\n=== EJEMPLOS DE SEGMENTACIÓN (MUESTRA ALEATORIA) ===")
    for idx in sample_indices:
        raw_text = text_list[idx]
        truncated_text = raw_text[:max_seq_length]  # Truncamos para no saturar

        # Procesar a bytes
        bytes_array = truncated_text.encode('utf-8')
        input_tensor = torch.tensor(list(bytes_array), dtype=torch.long).unsqueeze(0).to(device)

        # Obtener boundaries
        patch_boundaries_list = model.compute_patches(input_tensor)
        boundaries = patch_boundaries_list[0].cpu().numpy() if len(patch_boundaries_list) > 0 else []

        # Obtener entropías (opcional: entropía por posición)
        with torch.no_grad():
            ent_values = model.entropy_model(input_tensor, return_probabilities=False)
        ent_values = ent_values[0].cpu().numpy()

        # Calcular tamaños de parche
        patch_sizes = get_patch_sizes(list(boundaries), len(bytes_array))

        # Entropía promedio por parche
        patch_avg_entropy = []
        start = 0
        for end in boundaries:
            if end > start:
                avg_e = np.mean(ent_values[start:end])
                patch_avg_entropy.append(avg_e)
                start = end

        # Imprimir resultados
        print("-" * 70)
        print(f"Sample index: {idx}")
        display_text = truncated_text[:400] + ("..." if len(truncated_text) > 400 else "")
        print(f"Texto (trunc): {display_text!r}")
        print(f"Boundaries: {boundaries}")
        print(f"Tamaños de Parche: {patch_sizes}")
        print(f"Entropía Promedio x Parche: {[round(x, 3) for x in patch_avg_entropy]}")


# ==========================================================================
#            EVALUACIÓN DE GENERACIÓN (COHERENCIA / PERPLEJIDAD)
# ==========================================================================
def assess_generation_quality(
    model: BLT,
    text_list,
    device: torch.device,
    num_samples: int = 3,
    max_seq_length: int = 256,
    generation_length: int = 64
):
    """
    1) Toma unas muestras de texto, las divide en "prompt" (parte inicial)
       y "continuación real" (parte final).
    2) Genera texto usando `model` a partir del "prompt".
    3) Compara la generación con la "continuación real" midiendo Cross-Entropy/Perplejidad.
       - Para ello, calculamos logits del modelo contra la continuación real 
         y sacamos la CE Loss.
    4) Imprime resultados y ejemplos, indicando la calidad aproximada
       de la generación.

    Ojo: Esto no es un test exhaustivo de "coherencia semántica" en un sentido amplio,
    sino un proxy que indica si el modelo es capaz de continuar un texto
    de manera similar al real.
    """
    if not text_list:
        print("No hay texto para la evaluación de generación.")
        return

    print("\n=== EVALUACIÓN DE GENERACIÓN ===")
    # Seleccionar muestras aleatorias
    sample_indices = np.random.choice(len(text_list), size=min(num_samples, len(text_list)), replace=False)

    total_ce_loss = 0.0
    total_tokens = 0

    for idx in sample_indices:
        raw_text = text_list[idx]
        truncated_text = raw_text[: (2 * max_seq_length)]  # Doble del max_seq_length, por si se usa parte como prompt y parte como ref
        # Convertir a bytes
        full_bytes = list(truncated_text.encode('utf-8'))

        # Dividir en prompt + referencia
        # Suponemos un prompt inicial ~ la mitad, y la otra mitad "continuación real"
        half_point = max_seq_length // 2
        prompt_bytes = full_bytes[:half_point]
        ref_bytes = full_bytes[half_point : half_point + generation_length]  # Tomamos generation_length bytes de "continuación real"

        # 1) Generar
        with torch.no_grad():
            generated_seq = generate(
                model,
                start_bytes=prompt_bytes,
                max_length=half_point + generation_length,  # Generamos ~ generation_length más
                temperature=1.0,  # Ajustar si deseas
                top_k=50,
                patch_config=None,  # O se puede poner PatchingConfig(...) para ver su efecto
                device=device
            )
        # Generado en `generated_seq` son bytes (enteros)

        # 2) Calcular la Cross-Entropy entre logits(model(prompt + ref?), ref)
        #    - Haremos forward con (prompt + ref_bytes) y medimos CE en la parte de "ref_bytes".
        #    - Alternativamente, forward solo con prompt y comparamos? 
        #      Aquí elegimos la primera: (prompt+ref) para obtener la cross-entropy de la continuación. 
        input_tensor = torch.tensor(prompt_bytes + ref_bytes, dtype=torch.long).unsqueeze(0).to(device)
        target_tensor = torch.tensor(ref_bytes, dtype=torch.long).unsqueeze(0).to(device)

        # Pasamos al modelo. (Opcionalmente podríamos reusar patching adaptativo)
        # Retorna logits shape: [1, seq_len, vocab_size]
        logits = model(input_tensor)

        # Extraemos la porción de logits que corresponde a la parte "ref_bytes"
        # La parte de "ref_bytes" empieza en len(prompt_bytes)-1 (para predecir el primer token del ref)
        # y termina en (len(prompt_bytes + ref_bytes)-1).
        #   => Decimos "pred start" = len(prompt_bytes)
        # Sin embargo, para CrossEntropy, típicamente target está corrido en 1 (model shift).
        # Se simplifica si hacemos:
        pred_start = len(prompt_bytes)
        # Extraer la ventana con la longitud exacta de la referencia
        ref_len = len(ref_bytes)

        relevant_logits = logits[:, pred_start - 1 : pred_start - 1 + ref_len, :]
        # Ajuste: a veces se hace sin "shift". Pero probemos con shift:
        #   => relevant_logits = logits[:, pred_start:-1, :]  # etc.
        # Depende de cómo definimos target. 
        # Para no complicar, supongamos sin shift, e interpretamos "pred" en la misma posición.

        if relevant_logits.size(1) != ref_len:
            # Ajustar para no tener discrepancias
            # (Esto pasa si el prompt y la seguidilla se solapan en indices)
            min_len = min(relevant_logits.size(1), ref_len)
            relevant_logits = relevant_logits[:, :min_len, :]
            target_tensor = target_tensor[:, :min_len]

        # 3) Calcular CE
        logits_2d = relevant_logits.reshape(-1, relevant_logits.size(-1))
        target_1d = target_tensor.reshape(-1)
        ce_loss = torch.nn.functional.cross_entropy(logits_2d, target_1d, reduction='sum')
        total_ce_loss += ce_loss.item()
        total_tokens += target_1d.numel()

        # 4) Mostrar ejemplo
        # Convertir generated_seq a texto (cortamos a half_point + generation_length total, descartar si hay 0 token)
        # Notar que `generated_seq` puede ser mayor a half_point + generation_length. 
        # Tomamos la parte final generada en esos tokens.
        gen_text = bytes(generated_seq[len(prompt_bytes):]).decode('utf-8', errors='replace')
        ref_text = bytes(ref_bytes).decode('utf-8', errors='replace')
        prompt_text = bytes(prompt_bytes).decode('utf-8', errors='replace')

        print(f"\n[Ejemplo {idx}] Prompt (trunc): {prompt_text[:120]!r}...")
        print(f"Generado ({len(generated_seq) - len(prompt_bytes)} bytes): {gen_text!r}")
        print(f"Real   ({len(ref_bytes)} bytes): {ref_text!r}")

    # Promediar
    if total_tokens > 0:
        avg_ce = total_ce_loss / total_tokens
        ppl = np.exp(avg_ce)
        print(f"\n>>> Resultados de Generación (sobre {num_samples} muestras) <<<")
        print(f"Cross-Entropy promedio: {avg_ce:.4f}")
        print(f"Perplejidad promedio:   {ppl:.4f}")
    else:
        print("No se pudo calcular CE/PPL (sin tokens).")


# ==========================================================================
#                    CÁLCULO DE MÉTRICAS DE ENTROPÍA
# ==========================================================================
def compute_entropy_stats(
    model: BLT,
    dataloader: DataLoader,
    device: torch.device,
    max_sequence_length: int = 256
) -> dict:
    """
    Recorre el DataLoader y, para cada batch, obtiene:
      - Entropías (por posición)
      - patch_boundaries sugeridos
      - Métricas agregadas (promedio de entropía, etc.)
      - Tamaños de parche (distribución).
    """
    model.eval()

    all_entropies = []
    all_boundaries_counts = []
    all_patch_sizes_list = []

    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computando Entropías"):
            if batch.size(1) > max_sequence_length:
                batch = batch[:, :max_sequence_length]
            batch = batch.to(device)

            # 1) Obtener entropías
            entropies = model.entropy_model(batch, return_probabilities=False)
            if entropies.size(1) != batch.size(1):
                print(f"Warning: Entropies size mismatch. Expected {batch.size(1)}, got {entropies.size(1)}")
                continue

            # 2) Boundaries y patch sizes
            patch_boundaries_list = model.compute_patches(batch)

            # 3) Procesar resultados a nivel de cada secuencia
            for b_idx in range(batch.size(0)):
                seq_len = batch.shape[1]
                sample_ent = entropies[b_idx, :seq_len].cpu().numpy()
                all_entropies.append(sample_ent)

                boundaries = patch_boundaries_list[b_idx].cpu().numpy() if len(patch_boundaries_list) > 0 else []
                all_boundaries_counts.append(len(boundaries))

                # Calcular tamaños de parche
                patch_sizes = get_patch_sizes(boundaries, seq_len)
                all_patch_sizes_list.extend(patch_sizes)

                total_tokens += seq_len

    # 4) Calcular estadísticas de entropía
    entropies_conc = np.concatenate(all_entropies, axis=0) if len(all_entropies) > 0 else np.array([])
    stats = {
        "mean_entropy": float(entropies_conc.mean()) if entropies_conc.size > 0 else 0.0,
        "std_entropy": float(entropies_conc.std()) if entropies_conc.size > 0 else 0.0,
        "median_entropy": float(np.median(entropies_conc)) if entropies_conc.size > 0 else 0.0,
        "max_entropy": float(entropies_conc.max()) if entropies_conc.size > 0 else 0.0,
        "min_entropy": float(entropies_conc.min()) if entropies_conc.size > 0 else 0.0,
        "boundaries_mean": float(np.mean(all_boundaries_counts)) if all_boundaries_counts else 0.0,
        "boundaries_std": float(np.std(all_boundaries_counts)) if all_boundaries_counts else 0.0,
        "num_samples": len(all_entropies),
        "total_tokens": total_tokens
    }

    # 5) Calcular estadísticas de tamaños de parche
    patch_sizes_array = np.array(all_patch_sizes_list) if len(all_patch_sizes_list) > 0 else np.array([])
    patch_stats = {}
    if len(patch_sizes_array) > 0:
        patch_stats = {
            "mean_patch_size": float(patch_sizes_array.mean()),
            "std_patch_size": float(patch_sizes_array.std()),
            "median_patch_size": float(np.median(patch_sizes_array)),
            "min_patch_size": float(patch_sizes_array.min()),
            "max_patch_size": float(patch_sizes_array.max()),
        }
    else:
        patch_stats = {
            "mean_patch_size": 0.0,
            "std_patch_size": 0.0,
            "median_patch_size": 0.0,
            "min_patch_size": 0.0,
            "max_patch_size": 0.0,
        }

    # 6) Fusionar stats
    final_stats = {**stats, **patch_stats}

    return {
        "stats": final_stats,
        "all_entropies": all_entropies,
        "all_boundaries_counts": all_boundaries_counts,
        "all_patch_sizes": patch_sizes_array
    }

# ==========================================================================
#                      VISUALIZACIONES
# ==========================================================================
def visualize_entropy_distribution(entropies_list, output_dir="plots"):
    """
    Genera un histograma de la distribución de entropías e intenta guardar la figura.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    concatenated = np.concatenate(entropies_list, axis=0) if len(entropies_list) > 0 else np.array([])
    if concatenated.size == 0:
        print("No hay entropías para visualizar.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(concatenated, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Distribución de Entropías (por posición)")
    plt.xlabel("Entropía")
    plt.ylabel("Frecuencia")
    plt.grid(True)

    hist_path = os.path.join(output_dir, "entropy_distribution.png")
    plt.savefig(hist_path)
    print(f"Histograma de entropías guardado en: {hist_path}")
    plt.close()

def visualize_patch_boundaries(boundaries_counts, output_dir="plots"):
    """
    Histograma del número de parches por secuencia.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not boundaries_counts:
        print("No hay boundaries para visualizar.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(boundaries_counts, bins=20, color='green', alpha=0.7, edgecolor='black')
    plt.title("Conteo de Parches (boundaries) por Secuencia")
    plt.xlabel("Número de Parches")
    plt.ylabel("Frecuencia")
    plt.grid(True)

    hist_path = os.path.join(output_dir, "patch_boundaries_distribution.png")
    plt.savefig(hist_path)
    print(f"Histograma de conteo de parches guardado en: {hist_path}")
    plt.close()

def visualize_patch_size_distribution(patch_sizes, output_dir="plots"):
    """
    Histograma del tamaño de parches en general (en bytes).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if patch_sizes.size == 0:
        print("No hay tamaños de parche para visualizar.")
        return

    plt.figure(figsize=(10, 6))
    # Por ejemplo, usar intervalos de 4 en 4 (ajustar según dataset)
    plt.hist(patch_sizes, bins=range(0, patch_sizes.max() + 2, 4),
             color='purple', alpha=0.7, edgecolor='black')
    plt.title("Distribución de Tamaños de Parche")
    plt.xlabel("Tamaño de Parche (en bytes)")
    plt.ylabel("Frecuencia")
    plt.grid(True)

    hist_path = os.path.join(output_dir, "patch_size_distribution.png")
    plt.savefig(hist_path)
    print(f"Histograma de tamaños de parche guardado en: {hist_path}")
    plt.close()

# ==========================================================================
#                      FUNCIÓN PRINCIPAL
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluar y extender métricas de EntropyLM + test de generación en BLT.")
    parser.add_argument("--model_path", type=str, default="best_blt_model.pt",
                        help="Ruta al checkpoint del modelo entrenado.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Tamaño de lote para la evaluación.")
    parser.add_argument("--max_seq_length", type=int, default=256,
                        help="Longitud máxima de secuencia a evaluar.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cpu o cuda (si disponible).")
    parser.add_argument("--output_dir", type=str, default="plots",
                        help="Directorio donde se guardarán las gráficas.")
    parser.add_argument("--samples_to_inspect", type=int, default=0,
                        help="Número de secuencias que se imprimirán para ver segmentación real (modo debug).")
    parser.add_argument("--assess_generation", type=bool, default=False,
                        help="Si True, hará un test de generación comparativa.")
    parser.add_argument("--generation_samples", type=int, default=3,
                        help="Número de ejemplos para la prueba de generación.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) Cargar el modelo
    print(f"Cargando modelo desde: {args.model_path}")
    model, checkpoint = load_model(args.model_path, model_config)
    model = model.to(device)
    model.eval()

    # 2) Cargar datos de validación
    print("Cargando dataset de validación (openwebtext)...")
    val_size = 200000
    try:
        val_dataset_stream = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        data_list = []
        for i, example in enumerate(val_dataset_stream):
            if i >= val_size:
                break
            data_list.append(example["text"])
        print(f"Tamaño de validación (openwebtext): {len(data_list)}")
    except Exception as e:
        print(f"Error al cargar el dataset: {str(e)}")
        return

    # 3) Crear dataset y loader
    eval_dataset = ByteDataset(
        texts=data_list,
        max_length=args.max_seq_length,
        min_length=10,
        report_stats=False
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: collate_batch(x, max_length=args.max_seq_length),
        num_workers=2
    )

    # 4) Computar entropías y boundaries
    results = compute_entropy_stats(
        model=model,
        dataloader=eval_loader,
        device=device,
        max_sequence_length=args.max_seq_length
    )

    # 5) Imprimir estadísticas numéricas
    stats = results["stats"]
    print("\n=== Estadísticas Generales ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # 6) Visualizaciones
    print("\nGenerando visualizaciones...")
    visualize_entropy_distribution(results["all_entropies"], output_dir=args.output_dir)
    visualize_patch_boundaries(results["all_boundaries_counts"], output_dir=args.output_dir)
    visualize_patch_size_distribution(results["all_patch_sizes"], output_dir=args.output_dir)

    # 7) Opcional: ejemplos concretos de segmentación
    if args.samples_to_inspect > 0:
        analyze_text_segmentations(
            text_list=data_list,
            model=model,
            device=device,
            max_seq_length=args.max_seq_length,
            num_samples=args.samples_to_inspect
        )

    # 8) Opcional: evaluar la calidad de generación
    if args.assess_generation:
        assess_generation_quality(
            model=model,
            text_list=data_list,
            device=device,
            num_samples=args.generation_samples,
            max_seq_length=args.max_seq_length,
            generation_length=64  # se puede ajustar
        )


if __name__ == "__main__":
    main()
