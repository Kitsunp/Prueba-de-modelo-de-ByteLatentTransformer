# Prueba-de-modelo-de-ByteLatentTransformer
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

### Visualization
```python
from colorama import Fore, Style

print_metrics_table(metrics, "Validation Metrics")
```

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

### Common Issues
- CUDA out of memory
- Gradient overflow
- Attention mask compatibility
- Byte encoding errors

### Debug Mode
```python
# Enable debug logging
torch._dynamo.config.suppress_errors = True
print(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
