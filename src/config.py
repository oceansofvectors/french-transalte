"""
Configuration file for the translation model.
Contains all hyperparameters and settings.
"""

import torch
import os
from pathlib import Path

class Config:
    """Configuration class for the translation model."""

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
    VOCAB_SIZE = 10000  # Reduced from 30000 for faster training
    MIN_FREQ = 2
    MAX_LENGTH = 50  # Reduced from 100 for faster processing
    MAX_TRAIN_SAMPLES = None  # Set to a number (e.g., 50000) to limit training data

    # Special tokens
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    # Model architecture (optimized for speed)
    EMBEDDING_DIM = 128  # Reduced from 256
    HIDDEN_DIM = 256     # Reduced from 512 (becomes 512 with bidirectional)
    NUM_LAYERS = 1       # Reduced from 2
    DROPOUT = 0.2        # Reduced from 0.3
    BIDIRECTIONAL = True

    # Training
    BATCH_SIZE = 128     # Increased from 64 for faster throughput
    NUM_EPOCHS = 15      # Reduced from 20
    LEARNING_RATE = 0.001
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
    SAVE_EVERY = 1  # Save every N epochs
    PATIENCE = 5  # Early stopping patience

    # Logging
    LOG_EVERY = 100  # Log every N batches

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
        print("=" * 50)
