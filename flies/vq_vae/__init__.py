"""
VQ-VAE module with multiple quantization methods.

Available models:
- VQVAE: Standard VQ-VAE with GroupNorm pre-quantizer
- UnifiedVQVAE: Supports multiple quantization methods (VQ, FSQ, RVQ, LFQ)

Available quantizers:
- VectorQuantizer: Standard VQ implementation
- UnifiedQuantizer: Wrapper supporting multiple quantization methods
"""

from .vqvae import VQVAE, compute_strides_for_length
from .vqvae_unified import UnifiedVQVAE
from .quantizer import VectorQuantizer
from .unified_quantizer import UnifiedQuantizer
from .seq_encoder import SequenceEncoder
from .seq_decoder import SequenceDecoder

__all__ = [
    'VQVAE',
    'UnifiedVQVAE',
    'VectorQuantizer',
    'UnifiedQuantizer',
    'SequenceEncoder',
    'SequenceDecoder',
    'compute_strides_for_length',
]
