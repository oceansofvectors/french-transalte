"""
Training script for the translation model.
Supports MPS (Mac), CUDA, and CPU devices.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import sys
import logging
from datetime import datetime

from config import Config
from tokenizer import Tokenizer
from dataset import prepare_data
from model import Encoder, Decoder, Seq2Seq
from utils import (
    save_checkpoint,
    load_checkpoint,
    plot_losses,
    epoch_time,
    count_parameters,
    initialize_weights
)


def setup_logging(config):
    """
    Setup logging to both console and file.

    Args:
        config: Configuration object
    """
    # Create logs directory
    log_dir = config.OUTPUT_DIR
    log_dir.mkdir(exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def train_epoch(model, dataloader, optimizer, criterion, clip, device, config):
    """
    Train for one epoch.

    Args:
        model: The seq2seq model
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss criterion
        clip: Gradient clipping value
        device: Device to run on
        config: Configuration object

    Returns:
        Average epoch loss
    """
    model.train()
    epoch_loss = 0

    if config.USE_TPU:
        import torch_xla.core.xla_model as xm

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for i, (src, tgt, src_lengths, tgt_lengths) in enumerate(progress_bar):
        # Move to device
        src = src.to(device)
        tgt = tgt.to(device)
        src_lengths = src_lengths.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(src, src_lengths, tgt, config.TEACHER_FORCING_RATIO)

        # Reshape for loss calculation
        # output: [batch_size, tgt_len, output_dim]
        # tgt: [batch_size, tgt_len]

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)

        # Calculate loss (ignore padding)
        loss = criterion(output, tgt)

        # Backward pass
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Update weights (TPU-aware)
        if config.USE_TPU:
            xm.optimizer_step(optimizer)
            xm.mark_step()  # Mark step boundary for XLA
        else:
            optimizer.step()

        # Update loss (TPU-safe)
        if config.USE_TPU:
            # Reduce loss to CPU less frequently for TPU
            epoch_loss += loss.detach().cpu().item()
        else:
            epoch_loss += loss.item()

        # Update progress bar (less frequently on TPU)
        if not config.USE_TPU or i % 10 == 0:
            current_loss = loss.detach().cpu().item() if config.USE_TPU else loss.item()
            progress_bar.set_postfix({'loss': current_loss})

    return epoch_loss / len(dataloader)


def evaluate_epoch(model, dataloader, criterion, device, config=None):
    """
    Evaluate for one epoch.

    Args:
        model: The seq2seq model
        dataloader: Validation dataloader
        criterion: Loss criterion
        device: Device to run on
        config: Configuration object (optional)

    Returns:
        Average epoch loss
    """
    model.eval()
    epoch_loss = 0

    use_tpu = config.USE_TPU if config else False
    if use_tpu:
        import torch_xla.core.xla_model as xm

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

        for i, (src, tgt, src_lengths, tgt_lengths) in enumerate(progress_bar):
            # Move to device
            src = src.to(device)
            tgt = tgt.to(device)
            src_lengths = src_lengths.to(device)

            # Forward pass (no teacher forcing)
            output = model(src, src_lengths, tgt, 0)

            # Reshape for loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)

            # Calculate loss
            loss = criterion(output, tgt)

            # Update loss (TPU-safe)
            if use_tpu:
                epoch_loss += loss.detach().cpu().item()
            else:
                epoch_loss += loss.item()

            # Update progress bar (less frequently on TPU)
            if not use_tpu or i % 10 == 0:
                current_loss = loss.detach().cpu().item() if use_tpu else loss.item()
                progress_bar.set_postfix({'loss': current_loss})

    return epoch_loss / len(dataloader)


def train(config=None):
    """
    Main training function.

    Args:
        config: Configuration object (uses default if None)
    """
    # Use default config if none provided
    if config is None:
        config = Config

    # Create directories
    config.create_dirs()

    # Setup logging
    logger = setup_logging(config)

    # Print configuration
    logger.info("="*70)
    logger.info("Training Configuration")
    logger.info("="*70)
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Vocabulary Size: {config.VOCAB_SIZE}")
    logger.info(f"Embedding Dim: {config.EMBEDDING_DIM}")
    logger.info(f"Hidden Dim: {config.HIDDEN_DIM}")
    logger.info(f"Num Layers: {config.NUM_LAYERS}")
    logger.info(f"Dropout: {config.DROPOUT}")
    logger.info(f"Batch Size: {config.BATCH_SIZE}")
    logger.info(f"Learning Rate: {config.LEARNING_RATE}")
    logger.info(f"Num Epochs: {config.NUM_EPOCHS}")
    logger.info(f"Beam Width: {config.BEAM_WIDTH}")
    logger.info("="*70)

    # Initialize tokenizer
    logger.info("Initializing tokenizer and loading data...")
    tokenizer = Tokenizer(config)

    # Prepare data
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = prepare_data(
        config, tokenizer
    )

    # Get vocabulary sizes
    src_vocab_size, tgt_vocab_size = tokenizer.get_vocab_sizes()
    logger.info(f"Source vocabulary size: {src_vocab_size}")
    logger.info(f"Target vocabulary size: {tgt_vocab_size}")

    # Initialize model
    logger.info("Initializing model...")

    encoder = Encoder(
        input_dim=src_vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        use_packing=not config.USE_TPU  # Disable packing on TPU
    )

    decoder = Decoder(
        output_dim=tgt_vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )

    model = Seq2Seq(encoder, decoder, config.DEVICE).to(config.DEVICE)

    # Initialize weights
    initialize_weights(model)

    num_params = count_parameters(model)
    logger.info(f"Model has {num_params:,} trainable parameters")

    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)

    # Training loop
    logger.info("="*70)
    logger.info("Starting training...")
    logger.info("="*70)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Training...")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.GRAD_CLIP, config.DEVICE, config)

        # Evaluate
        logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Evaluating...")
        val_loss = evaluate_epoch(model, val_loader, criterion, config.DEVICE, config)

        end_time = time.time()

        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Calculate epoch time
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # Log progress
        logger.info(f"Epoch: {epoch+1:02}/{config.NUM_EPOCHS} | Time: {epoch_mins}m {epoch_secs}s")
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save checkpoint if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            logger.info(f"New best model! Val Loss: {val_loss:.4f}")
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                config.MODEL_DIR / 'best_model.pt'
            )
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{config.PATIENCE}")

        # Save periodic checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            logger.info(f"Saving periodic checkpoint at epoch {epoch+1}")
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                config.MODEL_DIR / f'checkpoint_epoch_{epoch+1}.pt'
            )

        # Early stopping
        if patience_counter >= config.PATIENCE:
            logger.warning(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Save final model
    logger.info("Saving final model...")
    save_checkpoint(
        model,
        optimizer,
        epoch,
        val_loss,
        config.MODEL_DIR / 'final_model.pt'
    )

    # Save tokenizer
    logger.info("Saving tokenizer...")
    tokenizer.save(config.MODEL_DIR / 'tokenizer.pkl')

    # Plot losses
    logger.info("Generating loss plots...")
    plot_losses(train_losses, val_losses, config.OUTPUT_DIR / 'training_loss.png')

    logger.info("="*70)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Total epochs: {epoch+1}")
    logger.info(f"Final train loss: {train_losses[-1]:.4f}")
    logger.info(f"Final val loss: {val_losses[-1]:.4f}")
    logger.info("="*70)


if __name__ == "__main__":
    # Train the model
    train()
