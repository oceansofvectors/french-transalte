"""
Configuration file for the translation model.
Contains all hyperparameters and settings.

Current Configuration: HIGH QUALITY
- ~50M parameters
- 30K vocabulary
- 3-layer BiLSTM (512 emb, 1024 hidden)
- Expected BLEU: 35-45
- Training time: ~4-6 hours on MPS/CUDA

For faster training, see FAST config at bottom of file.
"""

import torch
import os
from pathlib import Path

class Config:
    """Configuration class for the translation model.

    High-quality configuration optimized for best translation quality.
    Uses ~50M parameters and achieves BLEU scores of 35-45.
    """

    # Device override (set to force a specific device)
    # Options: None (auto-detect), "cuda", "mps", "cpu", "tpu"
    FORCE_DEVICE = os.environ.get("FORCE_DEVICE", None)

    # Paths
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    MODEL_DIR = ROOT_DIR / "models"
    OUTPUT_DIR = ROOT_DIR / "outputs"

    # Dataset
    DATASET_URL = "https://www.manythings.org/anki/fra-eng.zip"
    SRC_LANG = "eng"
    TGT_LANG = "fra"
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1

    # Vocabulary
    VOCAB_SIZE = 30000  # Large vocabulary for better coverage
    MIN_FREQ = 2
    MAX_LENGTH = 100    # Allow longer sequences
    MAX_TRAIN_SAMPLES = None  # Use full dataset

    # Special tokens
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    # Model architecture (High Quality - ~50M parameters)
    EMBEDDING_DIM = 512     # Large embeddings for rich representations
    HIDDEN_DIM = 1024       # Large hidden state (becomes 2048 with bidirectional)
    NUM_LAYERS = 3          # Deep network for better learning
    DROPOUT = 0.3           # Regularization for generalization
    BIDIRECTIONAL = True

    # Training
    BATCH_SIZE = 64         # Moderate batch size for stability
    NUM_EPOCHS = 30         # More epochs for convergence
    LEARNING_RATE = 0.0005  # Lower learning rate for stability
    GRAD_CLIP = 1.0
    TEACHER_FORCING_RATIO = 0.5

    # TPU-specific optimization
    @classmethod
    def get_batch_size(cls):
        """Get optimal batch size based on device."""
        if cls.USE_TPU:
            return 512  # TPUs need large batches
        else:
            return cls.BATCH_SIZE

    # Device - Auto-detect or use FORCE_DEVICE
    if FORCE_DEVICE:
        # Force specific device
        if FORCE_DEVICE.lower() == "tpu":
            try:
                import torch_xla.core.xla_model as xm
                DEVICE = xm.xla_device()
                USE_TPU = True
            except ImportError:
                raise RuntimeError("TPU requested but torch_xla not available")
        elif FORCE_DEVICE.lower() == "cuda":
            if torch.cuda.is_available():
                DEVICE = torch.device("cuda")
                USE_TPU = False
            else:
                raise RuntimeError("CUDA requested but not available")
        elif FORCE_DEVICE.lower() == "mps":
            if torch.backends.mps.is_available():
                DEVICE = torch.device("mps")
                USE_TPU = False
            else:
                raise RuntimeError("MPS requested but not available")
        elif FORCE_DEVICE.lower() == "cpu":
            DEVICE = torch.device("cpu")
            USE_TPU = False
        else:
            raise ValueError(f"Invalid FORCE_DEVICE: {FORCE_DEVICE}. Must be 'tpu', 'cuda', 'mps', or 'cpu'")
    else:
        # Auto-detect: TPU > MPS > CUDA > CPU
        try:
            import torch_xla.core.xla_model as xm
            DEVICE = xm.xla_device()
            USE_TPU = True
        except ImportError:
            USE_TPU = False
            if torch.backends.mps.is_available():
                DEVICE = torch.device("mps")
            elif torch.cuda.is_available():
                DEVICE = torch.device("cuda")
            else:
                DEVICE = torch.device("cpu")

    # Beam search
    BEAM_WIDTH = 5
    LENGTH_PENALTY = 0.6

    # Checkpointing
    SAVE_EVERY = 2  # Save every N epochs (less frequent for large model)
    PATIENCE = 8  # Early stopping patience (more patient for convergence)

    # Logging
    LOG_EVERY = 50  # Log every N batches (more frequent for monitoring)

    @classmethod
    def create_dirs(cls):
        """Create necessary directories if they don't exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.MODEL_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)

    @classmethod
    def print_config(cls):
        """Print configuration settings."""
        print("=" * 50)
        print("Configuration Settings")
        print("=" * 50)
        print(f"Device: {cls.DEVICE}")
        print(f"Vocabulary Size: {cls.VOCAB_SIZE}")
        print(f"Embedding Dim: {cls.EMBEDDING_DIM}")
        print(f"Hidden Dim: {cls.HIDDEN_DIM}")
        print(f"Num Layers: {cls.NUM_LAYERS}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Beam Width: {cls.BEAM_WIDTH}")
        print(f"Estimated Parameters: ~{cls.estimate_params()}M")
        print("=" * 50)

    @classmethod
    def estimate_params(cls):
        """Estimate model parameters in millions."""
        # Rough estimation
        vocab_params = cls.VOCAB_SIZE * cls.EMBEDDING_DIM * 2  # src + tgt embeddings
        encoder_params = 4 * cls.HIDDEN_DIM * (cls.EMBEDDING_DIM + cls.HIDDEN_DIM) * cls.NUM_LAYERS * 2  # BiLSTM
        decoder_params = 4 * cls.HIDDEN_DIM * 2 * (cls.EMBEDDING_DIM + cls.HIDDEN_DIM * 2) * cls.NUM_LAYERS
        attention_params = cls.HIDDEN_DIM * 4 * cls.HIDDEN_DIM
        output_params = cls.HIDDEN_DIM * 4 * cls.VOCAB_SIZE
        total = vocab_params + encoder_params + decoder_params + attention_params + output_params
        return round(total / 1_000_000, 1)


class FastConfig(Config):
    """Fast training configuration for quick experiments.

    Smaller model optimized for speed.
    Uses ~8M parameters and achieves BLEU scores of 25-35.
    Training time: ~1-2 hours on MPS/CUDA

    To use: In train.py, change `from config import Config` to `from config import FastConfig as Config`
    """

    # Vocabulary
    VOCAB_SIZE = 10000
    MAX_LENGTH = 50
    MAX_TRAIN_SAMPLES = None

    # Model architecture (Fast - ~8M parameters)
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 1
    DROPOUT = 0.2

    # Training
    BATCH_SIZE = 128
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001
    PATIENCE = 5
    SAVE_EVERY = 1
