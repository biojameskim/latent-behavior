# VQ-VAE Docs

## Overview

VQ-VAE (Vector Quantized Variational Autoencoder) learns to represent temporal sequences using a discrete codebook of learned embeddings. For fly behavior analysis, this discovers a vocabulary of behavior "syllables" that can be combined to describe complex behavioral sequences.

**Architecture**: Input → Encoder → Vector Quantizer → Decoder → Reconstructed Output

## Implementation Components

### 1. Encoder (`seq_encoder.py`)

Compresses temporal sequences into latent representations using 1D convolutions.

**Architecture**:
- Downsampling blocks: Conv1d (kernel=4, stride=2) → ReLU → BatchNorm
- Residual blocks (2 per layer by default) for learning complex temporal patterns
- Final projection to embedding dimension

**Input**: `(batch, input_dim, sequence_length)` - e.g., (B, 48, 150) for 24 keypoints × 2 coords
**Output**: `(batch, embedding_dim, reduced_length)` - e.g., (B, 512, 18) with 3 hidden layers

**Key feature**: Each downsampling layer reduces sequence length by 2x while increasing feature channels, trading temporal resolution for richer feature representations.

### 2. Vector Quantizer (`quantizer.py`)

The core VQ-VAE component that maps continuous encoder outputs to discrete codebook entries.

**Codebook**: Learnable embedding table with `num_embeddings` entries (e.g., 512 behavior syllables)

**Process**:
1. For each encoder output, find nearest codebook entry via L2 distance
2. Replace continuous values with discrete codebook vectors
3. Use straight-through estimator to allow gradient flow

**Losses**:
- **Codebook loss**: Moves codebook entries closer to encoder outputs
- **Commitment loss**: Encourages encoder to commit to codebook entries
- Total: `loss = codebook_loss + β × commitment_loss` (β = 0.25 default)

**Outputs**:
- `z_q`: Quantized embeddings
- `vq_loss`: VQ loss for training
- `perplexity`: Codebook usage metric (higher = more codes actively used)
- `encoding_indices`: Which codebook entry was selected at each timestep

### 3. Decoder (`seq_decoder.py`)

Reconstructs sequences from quantized latent representations, mirroring the encoder architecture.

**Architecture**:
- Residual blocks (2 per layer) before upsampling
- Upsampling blocks: ConvTranspose1d (kernel=4, stride=2) → ReLU → BatchNorm
- Final upsampling to original input dimension

**Input**: `(batch, embedding_dim, reduced_length)` - quantized latent codes
**Output**: `(batch, output_dim, sequence_length)` - reconstructed sequence

### 4. Residual Block (`residual.py`)

Standard ResNet-style residual block adapted for 1D temporal data.

**Architecture**: Conv1d → BatchNorm → ReLU → Conv1d → BatchNorm → (Add skip connection) → ReLU

**Purpose**:
- Allows deeper networks without vanishing gradients
- Helps learn complex temporal patterns
- Maintains sequence length (padding preserves dimensions)

### 5. Complete VQ-VAE (`vqvae.py`)

Ties all components together into a complete model.

**Main methods**:
- `forward(x)`: Full pass through encoder → quantizer → decoder
- `encode(x)`: Extract discrete behavior codes from sequences
- `decode_codes(indices)`: Generate sequences from discrete codes

**Training**:
- Reconstruction loss: MSE between input and reconstruction
- VQ loss: Codebook + commitment loss
- Total loss: `reconstruction_loss + vq_loss`

## Example Usage

```python
from vq_vae.vqvae import VQVAE

# Initialize model for fly data (24 keypoints × 2 coordinates = 48 features)
model = VQVAE(
    input_dim=48,
    hidden_dims=[64, 128, 256],      # 3 layers → 8x temporal compression
    embedding_dim=512,                # Latent dimension
    num_embeddings=512,               # 512 behavior syllables in codebook
    sequence_length=150,              # Input sequence length (window size)
    num_residual_blocks=2,            # 2 residual blocks per layer
    commitment_cost=0.25              # β for commitment loss
)

# Training forward pass
x_recon, vq_loss, perplexity, encodings, encoding_indices = model(x)
recon_loss = F.mse_loss(x_recon, x)
total_loss = recon_loss + vq_loss

# Extract behavior codes for analysis
codes = model.encode(x)  # (batch, reduced_time_steps)
# codes[i, t] gives the behavior syllable ID at timestep t

# Generate sequences from codes
x_generated = model.decode_codes(codes)
```

## Key Hyperparameters

- **num_embeddings**: Codebook size (e.g., 512) - number of discrete behavior patterns
- **embedding_dim**: Latent dimension (e.g., 512) - richness of each code representation
- **sequence_length**: Input sequence length (e.g., 150 frames) - must match window size
  - **Important**: Choose values with many factors for auto-stride computation
  - Good choices: 120, 144, 150, 160, 192, 200 (highly composite numbers)
  - Avoid: Prime numbers (97, 101, 127) or values with few factors
  - The model will raise a clear error if the sequence length is incompatible
- **hidden_dims**: Encoder/decoder layer sizes - controls model capacity
  - More layers = more temporal compression but requires more factors in sequence_length
- **num_residual_blocks**: Residual blocks per layer (2 is standard) - deeper = more complex patterns
- **commitment_cost**: β weight (0.25 typical) - balance between codebook and encoder updates

## Architecture Decisions

### Why variable strides?
- Strides are automatically computed to evenly divide the sequence length
- For 150 frames with 3 layers: strides=[5,3,2] → 150→30→10→5 timesteps
- This ensures exact reconstruction (no padding/cropping artifacts)
- Trades temporal resolution for richer feature representations
- More flexible than fixed stride=2 (works with any sequence length that has factors)

### Why residual blocks?
- Allows deeper networks to learn complex temporal patterns
- Prevents vanishing gradients
- Fly behaviors have subtle, hierarchical temporal structure that benefits from depth

### Why discrete codes?
- **Interpretability**: Each code represents a behavior syllable
- **Compression**: Long sequences → short discrete code sequences
- **Compositionality**: Complex behaviors = sequences of simple syllables
- **Analysis**: Can analyze behavior as "grammar" of syllables

## Troubleshooting

### ValueError: "Cannot create N downsampling layers for sequence_length=X"

**Cause**: Your chosen `sequence_length` (window_size) doesn't have enough prime factors to distribute across all encoder/decoder layers.

**Example**:
- `sequence_length=97` (prime number) with 3 layers fails
- `sequence_length=150` (= 2×3×5²) with 3 layers works → strides=[5,3,2]

**Solutions**:
1. **Use a different window_size** with more factors:
   - Good: 120 (2³×3×5), 144 (2⁴×3²), 150 (2×3×5²), 160 (2⁵×5), 192 (2⁶×3), 200 (2³×5²)
   - Bad: 97, 101, 127 (primes), 98 (2×7²)

2. **Reduce the number of hidden layers**:
   ```python
   # Instead of 3 layers:
   hidden_dims=[64, 128, 256]  # Needs 3 factors

   # Use 2 layers:
   hidden_dims=[128, 256]  # Needs only 2 factors
   ```

3. **Manually specify strides** (advanced):
   ```python
   model = VQVAE(
       sequence_length=100,
       hidden_dims=[64, 128],
       strides=[5, 4]  # 100→20→5 (manually chosen)
   )
   ```

### Decoder output length mismatch

If you see this error, it indicates a bug in stride computation. Please report it as an issue.

The variable stride system should guarantee exact reconstruction for any compatible sequence length.

## Resources
- Google's original implementation (in Tensorflow) [[Github]](https://github.com/google-deepmind/sonnet/blob/v1/sonnet/python/modules/nets/vqvae.py)
- VQ-VAE Pytorch implementation [[Github]](https://github.com/MishaLaskin/vqvae)
- Vector (and Scalar) Quantization (in Pytorch) [[Github]](https://github.com/lucidrains/vector-quantize-pytorch)
- VQ-MAP [[OpenReview]](https://openreview.net/forum?id=uiFuqvkpAt) [[Github]](https://github.com/tqxli/vqmap)