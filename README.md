# BLT (Byte-Level Transformer)

A PyTorch implementation of a Byte-Level Transformer model for efficient text generation and processing at the byte level with adaptive patching mechanisms.

## Features

- Byte-level processing with n-gram enhanced embeddings
- Adaptive patching with entropy-based boundary detection
- Hierarchical architecture with local and global processing
- Memory-efficient implementation without caching mechanisms
- Support for multiple patching schemes: entropy-based, fixed-stride, and space-based
- Advanced attention mechanisms with rotary positional embeddings
- Robust training and generation utilities
- Dynamic batch handling and memory optimization
- Multi-head attention with lambda-based combination
- Hierarchical memory with adaptive gating
- Entropy-based patch boundary detection
- N-gram enhanced byte embeddings
- RMS normalization for improved stability

## Requirements

```bash
torch>=2.0.0
numpy>=1.21.0
tqdm>=4.62.0
datasets>=2.0.0
colorama>=0.4.4
```

## Installation and Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd blt-transformer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
# Basic model initialization and training
from blt_model import BLTConfig, BLT, train_model
from train import main

# Run the main training script
if __name__ == "__main__":
    main()
```

## Detailed Model Architecture

### 1. Core Components

#### Local Encoder
- Processes raw byte inputs
- N-gram enhanced embeddings
- Multiple encoder layers with self-attention
- RMS normalization and residual connections

#### Global Transformer
- Handles patch-level processing
- Adaptive attention mechanisms
- Hierarchical memory integration
- Lambda-based head combination

#### Local Decoder
- Reconstructs byte sequences
- Cross-attention with global context
- Multiple decoder layers
- Output projection to byte space

### 2. Attention Mechanisms

#### Multi-Head Attention
- Rotary positional embeddings
- Lambda-based head combination
- Headwise normalization
- Efficient attention patterns

#### Cross-Attention
- Connects local and global processors
- Adaptive attention weights
- Context integration

### 3. Patching System

#### Entropy-Based Patching
```python
patch_config = PatchingConfig(
    scheme='entropy',
    entropy_threshold=0.5,
    use_monotonic=True
)
```

#### Fixed-Stride Patching
```python
patch_config = PatchingConfig(
    scheme='fixed',
    stride=128,
    reset_context=True
)
```

## Model Configuration

### Basic Configuration
```python
config = BLTConfig(
    hidden_size=512,
    intermediate_size=2048,
    num_heads=16,
    encoder_layers=2,
    global_layers=8,
    decoder_layers=6,
    attention_dropout=0.12,
    resid_dropout=0.12,
    ngram_vocab_size=150000,
    window_size=512,
    max_position_embeddings=4096
)
```

### Advanced Configuration
```python
config = BLTConfig(
    hidden_size=768,
    intermediate_size=3072,
    num_heads=24,
    encoder_layers=4,
    global_layers=12,
    decoder_layers=8,
    attention_dropout=0.1,
    resid_dropout=0.1,
    ngram_vocab_size=200000,
    window_size=1024,
    max_position_embeddings=8192,
    entropy_model_layers=3,
    entropy_context_size=1024,
    entropy_threshold=0.6,
    min_patch_size=64,
    max_patch_size=1024,
    initial_entropy_threshold=0.5
)
```

## Training Process

### 1. Data Preparation
```python
from blt_model import ByteDataset
from torch.utils.data import DataLoader

# Create datasets
train_dataset = ByteDataset(
    texts=train_texts,
    max_length=config['max_sequence_length'],
    min_length=config['min_text_length']
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    collate_fn=collate_batch
)
```

### 2. Training Configuration
```python
training_config = {
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'warmup_steps': 500,
    'max_grad_norm': 2.0,
    'gradient_accumulation_steps': 4,
    'max_sequence_length': 3072,
    'batch_size': 32,
    'eval_batch_size': 32,
    'patience': 3,
    'num_epochs': 3,
    'min_text_length': 30
}
```

### 3. Training Loop
```python
model, best_val_loss = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=training_config,
    num_epochs=training_config['num_epochs']
)
```

## Interactive Usage

### 1. Loading a Trained Model
```python
model_path = 'best_blt_model.pt'
model, checkpoint = load_model(model_path, model_config)
```

### 2. Text Generation
```python
patch_config = PatchingConfig(
    scheme='entropy',
    entropy_threshold=0.5,
    use_monotonic=True
)

response = generate_text(
    model=model,
    start_text=user_input,
    max_length=500,
    temperature=1.0,
    patch_config=patch_config,
    top_k=50
)
```

## Performance Optimization

### Memory Efficiency
- Dynamic batch sizing
- Gradient accumulation
- Efficient attention patterns
- Adaptive patch sizing

### Training Stability
- RMS normalization
- Residual connections
- Lambda-based head combination
- Hierarchical memory

## Model Evaluation

### Metrics
```python
# Calculate bits per byte
bpb = compute_bits_per_byte(model, validation_data, patch_config)

# Generate evaluation metrics
metrics = {
    'loss': val_results['loss'],
    'accuracy': val_results['accuracy'],
    'perplexity': val_results['perplexity'],
    'confidence_mean': val_results['confidence_mean']
}
```
## Training and Validation Metrics Log
### Información General
```plaintext
───────────────────────────────
Tiempo de época               00:24:06
Tiempo total                  00:24:06
Learning Rate                 9.62e-05
Memoria GPU (GB)              19.9
───────────────────────────────

### Métricas de Entrenamiento
──────────────────────────────
Pérdida                       3.0658
Accuracy                      0.5810
Perplejidad                   21.4511
Confianza Media               0.3877
Confianza Mín                 0.0
Confianza Máx                 1.0
──────────────────────────────

### Métricas de Validación
──────────────────────────────
Pérdida                       0.6306
Accuracy                      0.8918
Perplejidad                   1.8788
Confianza Media               0.8182102
Confianza Mín                 8.019899e-13
Confianza Máx                 0.99999726
──────────────────────────────

✓ Guardado nuevo mejor modelo con pérdida de validación: 0.6306

=================================================================================================================================
                                                            Época 2/5
=================================================================================================================================

Información General
───────────────────────────────
Tiempo de época               00:23:32
Tiempo total                  00:48:03
Learning Rate                 8.55e-05
Memoria GPU (GB)              19.9
───────────────────────────────

Métricas de Entrenamiento
──────────────────────────────
Pérdida                       2.0842
Accuracy                      0.8181
Perplejidad                   8.0385
Confianza Media               0.5835
Confianza Mín                 0.0
Confianza Máx                 1.0
──────────────────────────────

Métricas de Validación
──────────────────────────────
Pérdida                       0.3664
Accuracy                      0.9366
Perplejidad                   1.4425
Confianza Media               0.91945153
Confianza Mín                 1.689753e-14
Confianza Máx                 0.9999758
──────────────────────────────

✓ Guardado nuevo mejor modelo con pérdida de validación: 0.3664

=================================================================================================================================
                                                            Época 3/5
=================================================================================================================================

Información General
───────────────────────────────
Tiempo de época               00:23:33
Tiempo total                  01:12:05
Learning Rate                 6.95e-05
Memoria GPU (GB)              19.9
───────────────────────────────

Métricas de Entrenamiento
──────────────────────────────
Pérdida                       1.8699
Accuracy                      0.8741
Perplejidad                   6.4874
Confianza Media               0.654
Confianza Mín                 0.0
Confianza Máx                 1.0
──────────────────────────────

Métricas de Validación
──────────────────────────────
Pérdida                       0.2585
Accuracy                      0.9532
Perplejidad                   1.2950
Confianza Media               0.94478035
Confianza Mín                 2.4185464e-14
Confianza Máx                 0.9999392
──────────────────────────────

✓ Guardado nuevo mejor modelo con pérdida de validación: 0.2585

## Entropies stats
=== Estadísticas Generales ===
mean_entropy: 3.6470255851745605
std_entropy: 1.292510986328125
median_entropy: 3.430821180343628
max_entropy: 20.774738311767578
min_entropy: 0.36849021911621094
boundaries_mean: 7.000005
boundaries_std: 0.0022360623873228583
num_samples: 200000
total_tokens: 51200000

## Training and Validation Data

The model supports multiple data sources:

### Default Datasets
- OpenWebText for training (110,000 examples)
- Wikitext-103 for validation (100,000 examples)

### Custom Dataset Usage
```python
custom_dataset = ByteDataset(
    texts=your_texts,
    max_length=3072,
    min_length=30,
    report_stats=True
)
```

## Error Handling and Debugging

### Common Issues and Limitations

#### Technical Issues
- CUDA out of memory
- Gradient overflow
- Attention mask compatibility
- Byte encoding errors

#### Performance Limitations
- Lack of optimization for vectorized operations
- Incomplete adaptability in certain scenarios
- Learning difficulty (requires large amounts of data for text generation)
- Code modularity challenges
- Cache implementation problems
- Low-level CUDA kernel optimization issues
- Low-level optimization challenges in general performance

#### Resource Requirements
- High computational requirements for training
- Significant memory usage for larger models
- Extended training time for optimal performance

### Debug Mode
```python
# Enable debug logging
torch._dynamo.config.suppress_errors = True
print(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
```

## Model Capabilities and Future Development

### Demonstrated Capabilities
- Text Reconstruction: The model has demonstrated strong capability in accurate text reconstruction, being able to regenerate input text with high fidelity
- Rapid Adaptation: Shows quick adaptation capabilities (implemented through extensive dropout layers)
- Byte-Level Understanding: Demonstrates deep understanding of byte-level patterns and structures

### Current Development
- Native Audio Processing: Currently testing capabilities for native audio decoding and generation
- Multimodal Extensions: Working on expanding the model's capabilities to handle multiple modalities
- Spanish Documentation: Full documentation available in Spanish
- Enhanced Features: Planning to release complete codebase with improved multimodal capabilities

### Potential Quick Improvements (Experimental)
Note: The following improvements are theoretical and haven't been tested, but could be relatively quick to implement:

#### Architecture Enhancements
- Dynamic patch size adjustment based on content complexity
- Adaptive lambda parameter scaling
- Hierarchical memory pooling mechanism
- Progressive dimensionality reduction in attention
- Conditional computation paths for different modalities

#### Performance Optimizations
- Fused kernel operations for attention mechanisms
- Sparse attention patterns for long sequences
- Memory-efficient gradient accumulation
- Adaptive batch sizing based on sequence length
- Dynamic precision switching for different operations

#### Training Improvements
- Curriculum learning for patch size progression
- Multi-task pretraining objectives
- Dynamic temperature scaling during generation
- Adaptive entropy thresholding
- Progressive layer freezing during fine-tuning

#### Multimodal Extensions (Experimental)
- Byte-level audio compression integration
- Image patch encoding at byte level
- Cross-modal attention mechanisms
- Unified byte representation for different modalities
- Modality-specific embedding layers

#### Technical Optimizations
- Custom CUDA kernels for n-gram processing
- Optimized memory layout for attention patterns
- Efficient cache implementation strategies
- Stream-based processing for long sequences
- Parallel patch boundary detection

Note: These improvements are speculative and would require testing for effectiveness and compatibility.

## Additional Notes
- The extensive use of dropout layers is intentional, supporting the model's rapid adaptation capabilities
- The byte-level approach shows promise for extending beyond text to other modalities
- Future releases will include extended multimodal capabilities and improvements

## Model Foundation and References

This implementation draws inspiration from several groundbreaking papers in the field:

### Primary References

1. **DIFFERENTIAL TRANSFORMER**
   - Authors: Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, Furu Wei
   - Institution: Microsoft Research & Tsinghua University
   - Paper: arXiv:2410.05258v1 [cs.CL] 7 Oct 2024
   - URL: https://aka.ms/GeneralAI

2. **ByteLatentTransformer: Patches Scale Better Than Tokens**
   - Authors: Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, et al.
   - Institution: FAIR at Meta, University of Washington, University of Chicago
   - Notable features adapted:
     - Patch-based processing
     - Byte-level representations
     - Latent space transformations

Note: Additional foundational papers and architectural influences will be documented in future updates.

### Key Architectural Influences
- Differential computation mechanisms from DIFFERENTIAL TRANSFORMER
- Patch-based scaling strategies from ByteLatentTransformer
- Byte-level processing techniques
- Attention mechanism adaptations
- Memory-efficient design patterns

## Real-Time Implementation Challenges

The model currently faces several challenges for real-time applications:

### Hardware-Specific Limitations
- GPU optimization requirements
- CPU performance bottlenecks
- Cross-architecture compatibility issues
- Memory bandwidth constraints

### Computation Optimization Needs
- Real-time inference latency
- Cross-platform performance variability
- Resource utilization efficiency
- Batch processing overhead

### Future Optimization Goals
- Hardware-agnostic performance improvements
- Reduced inference latency
- Better resource utilization
- Enhanced cross-platform compatibility
