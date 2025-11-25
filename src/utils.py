"""
Utility functions for the translation model.
"""

import torch
import matplotlib.pyplot as plt
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model checkpoint.

    Args:
        model: The model to save
        optimizer: The optimizer
        epoch: Current epoch
        loss: Current loss
        path: Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device):
    """
    Load model checkpoint.

    Args:
        model: The model to load into
        optimizer: The optimizer to load into
        path: Path to the checkpoint
        device: Device to load the model on

    Returns:
        epoch: The epoch from the checkpoint
        loss: The loss from the checkpoint
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    logger.info(f"Checkpoint loaded from {path} (epoch {epoch}, loss {loss:.4f})")
    return epoch, loss


def plot_losses(train_losses, val_losses, save_path):
    """
    Plot training and validation losses.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Loss plot saved to {save_path}")


def epoch_time(start_time, end_time):
    """
    Calculate elapsed time.

    Args:
        start_time: Start time
        end_time: End time

    Returns:
        elapsed_mins: Elapsed minutes
        elapsed_secs: Elapsed seconds
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model: The model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model):
    """
    Initialize model weights using Xavier initialization.

    Args:
        model: The model to initialize
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            torch.nn.init.xavier_normal_(param.data)
        elif 'bias' in name:
            torch.nn.init.constant_(param.data, 0)


def calculate_accuracy(predictions, targets, pad_idx):
    """
    Calculate accuracy, ignoring padding tokens.

    Args:
        predictions: Predicted token indices [batch_size, seq_len]
        targets: Target token indices [batch_size, seq_len]
        pad_idx: Padding token index

    Returns:
        Accuracy as a percentage
    """
    # Get predictions by taking argmax
    if predictions.dim() == 3:
        predictions = predictions.argmax(dim=-1)

    # Create mask for non-padding tokens
    non_pad_mask = (targets != pad_idx)

    # Calculate correct predictions
    correct = (predictions == targets) & non_pad_mask
    accuracy = correct.sum().item() / non_pad_mask.sum().item()

    return accuracy * 100


def print_sample_translations(model, dataset, tokenizer, device, num_samples=5):
    """
    Print sample translations from the model.

    Args:
        model: The trained model
        dataset: Dataset to sample from
        tokenizer: Tokenizer object
        device: Device to run on
        num_samples: Number of samples to print
    """
    from .beam_search import beam_search

    model.eval()
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            src, tgt = dataset[i]
            src_tensor = torch.tensor(src).unsqueeze(0).to(device)

            # Get translation using beam search
            predicted_tokens = beam_search(
                model,
                src_tensor,
                max_length=100,
                beam_width=5,
                device=device
            )

            # Convert tokens to words
            src_words = tokenizer.decode(src, is_target=False)
            tgt_words = tokenizer.decode(tgt, is_target=True)
            pred_words = tokenizer.decode(predicted_tokens, is_target=True)

            print(f"\nSample {i+1}:")
            print(f"Source:      {src_words}")
            print(f"Target:      {tgt_words}")
            print(f"Predicted:   {pred_words}")
