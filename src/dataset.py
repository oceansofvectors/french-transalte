"""
Dataset loading and preprocessing for English-French translation.
Downloads and processes the Tatoeba dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import requests
import zipfile
import io
import random
from tqdm import tqdm


class TranslationDataset(Dataset):
    """Dataset for translation pairs."""

    def __init__(self, src_sentences, tgt_sentences):
        """
        Initialize the dataset.

        Args:
            src_sentences: List of source (English) sentence token indices
            tgt_sentences: List of target (French) sentence token indices
        """
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        return self.src_sentences[idx], self.tgt_sentences[idx]


def download_tatoeba_dataset(data_dir):
    """
    Download the Tatoeba English-French dataset.

    Args:
        data_dir: Directory to save the dataset

    Returns:
        Path to the downloaded file
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    file_path = data_dir / "fra.txt"

    if file_path.exists():
        print(f"Dataset already exists at {file_path}")
        return file_path

    print("Downloading Tatoeba English-French dataset...")
    url = "https://www.manythings.org/anki/fra-eng.zip"

    # Add headers to mimic a browser request (website blocks bots)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            zip_file.extractall(data_dir)

        print(f"Dataset downloaded and extracted to {data_dir}")
        return file_path

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise


def load_data(file_path, max_length=100):
    """
    Load and parse the dataset file.

    Args:
        file_path: Path to the data file
        max_length: Maximum sentence length (longer sentences are filtered)

    Returns:
        List of (English, French) sentence pairs
    """
    print(f"Loading data from {file_path}...")
    pairs = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                eng = parts[0].strip()
                fra = parts[1].strip()

                # Filter by length
                if len(eng.split()) <= max_length and len(fra.split()) <= max_length:
                    pairs.append((eng, fra))

    print(f"Loaded {len(pairs)} sentence pairs")
    return pairs


def split_data(pairs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split data into train, validation, and test sets.

    Args:
        pairs: List of sentence pairs
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed for reproducibility

    Returns:
        train_pairs, val_pairs, test_pairs
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    print(f"Data split: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")

    return train_pairs, val_pairs, test_pairs


def collate_fn(batch, pad_idx):
    """
    Collate function for batching.
    Pads sequences to the same length within a batch.

    Args:
        batch: List of (src, tgt) pairs
        pad_idx: Padding token index

    Returns:
        src_batch: Padded source sequences [batch_size, max_src_len]
        tgt_batch: Padded target sequences [batch_size, max_tgt_len]
        src_lengths: Original lengths of source sequences
        tgt_lengths: Original lengths of target sequences
    """
    src_batch, tgt_batch = [], []

    for src, tgt in batch:
        src_batch.append(torch.tensor(src, dtype=torch.long))
        tgt_batch.append(torch.tensor(tgt, dtype=torch.long))

    # Pad sequences
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=pad_idx, batch_first=True)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=pad_idx, batch_first=True)

    # Get lengths
    src_lengths = torch.tensor([len(src) for src, _ in batch], dtype=torch.long)
    tgt_lengths = torch.tensor([len(tgt) for _, tgt in batch], dtype=torch.long)

    return src_batch, tgt_batch, src_lengths, tgt_lengths


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, pad_idx):
    """
    Create DataLoaders for train, validation, and test sets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        pad_idx: Padding token index

    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx)
    )

    return train_loader, val_loader, test_loader


def prepare_data(config, tokenizer):
    """
    Complete data preparation pipeline.

    Args:
        config: Configuration object
        tokenizer: Tokenizer object

    Returns:
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    """
    # Download dataset
    file_path = download_tatoeba_dataset(config.DATA_DIR)

    # Load data
    pairs = load_data(file_path, max_length=config.MAX_LENGTH)

    # Limit training data if specified (for faster training/testing)
    if config.MAX_TRAIN_SAMPLES is not None and len(pairs) > config.MAX_TRAIN_SAMPLES:
        print(f"Limiting dataset to {config.MAX_TRAIN_SAMPLES} samples (from {len(pairs)})")
        pairs = pairs[:config.MAX_TRAIN_SAMPLES]

    # Split data
    train_pairs, val_pairs, test_pairs = split_data(
        pairs,
        train_ratio=config.TRAIN_SPLIT,
        val_ratio=config.VAL_SPLIT,
        test_ratio=config.TEST_SPLIT
    )

    # Build vocabulary
    print("Building vocabulary...")
    tokenizer.build_vocab(train_pairs)

    # Tokenize data
    print("Tokenizing data...")
    train_src, train_tgt = tokenizer.tokenize_pairs(train_pairs)
    val_src, val_tgt = tokenizer.tokenize_pairs(val_pairs)
    test_src, test_tgt = tokenizer.tokenize_pairs(test_pairs)

    # Create datasets
    train_dataset = TranslationDataset(train_src, train_tgt)
    val_dataset = TranslationDataset(val_src, val_tgt)
    test_dataset = TranslationDataset(test_src, test_tgt)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        config.BATCH_SIZE, config.PAD_IDX
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
