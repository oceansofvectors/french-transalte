"""
Evaluation script for the translation model.
Calculates BLEU score and shows sample translations.
"""

import torch
from tqdm import tqdm
from sacrebleu import corpus_bleu

from config import Config
from tokenizer import Tokenizer
from dataset import prepare_data
from model import Encoder, Decoder, Seq2Seq
from beam_search import beam_search
from utils import load_checkpoint


def translate_dataset(model, dataset, tokenizer, device, config, use_beam_search=True):
    """
    Translate all sentences in a dataset.

    Args:
        model: Trained model
        dataset: Dataset to translate
        tokenizer: Tokenizer object
        device: Device to run on
        config: Configuration object
        use_beam_search: Whether to use beam search (vs greedy)

    Returns:
        predictions: List of predicted sentences (strings)
        references: List of reference sentences (strings)
    """
    model.eval()

    predictions = []
    references = []

    with torch.no_grad():
        for src, tgt in tqdm(dataset, desc="Translating"):
            # Prepare source
            src_tensor = torch.tensor(src).unsqueeze(0).to(device)

            # Translate
            if use_beam_search:
                predicted_tokens = beam_search(
                    model,
                    src_tensor,
                    max_length=config.MAX_LENGTH,
                    beam_width=config.BEAM_WIDTH,
                    device=device,
                    sos_idx=config.SOS_IDX,
                    eos_idx=config.EOS_IDX,
                    length_penalty=config.LENGTH_PENALTY
                )
            else:
                from beam_search import greedy_decode
                predicted_tokens = greedy_decode(
                    model,
                    src_tensor,
                    max_length=config.MAX_LENGTH,
                    device=device,
                    sos_idx=config.SOS_IDX,
                    eos_idx=config.EOS_IDX
                )

            # Decode
            predicted_sentence = tokenizer.decode_sentence(predicted_tokens, is_target=True)
            reference_sentence = tokenizer.decode_sentence(tgt, is_target=True)

            predictions.append(predicted_sentence)
            references.append(reference_sentence)

    return predictions, references


def calculate_bleu(predictions, references):
    """
    Calculate BLEU score.

    Args:
        predictions: List of predicted sentences
        references: List of reference sentences

    Returns:
        BLEU score
    """
    # SacreBLEU expects references as a list of lists
    references = [[ref] for ref in references]

    bleu = corpus_bleu(predictions, references)
    return bleu.score


def show_sample_translations(predictions, references, num_samples=10):
    """
    Show sample translations.

    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        num_samples: Number of samples to show
    """
    print("\n" + "="*70)
    print("Sample Translations")
    print("="*70)

    for i in range(min(num_samples, len(predictions))):
        print(f"\nSample {i+1}:")
        print(f"  Reference:  {references[i]}")
        print(f"  Prediction: {predictions[i]}")


def evaluate(model_path=None, use_beam_search=True, config=None):
    """
    Main evaluation function.

    Args:
        model_path: Path to the model checkpoint (uses best model if None)
        use_beam_search: Whether to use beam search
        config: Configuration object (uses default if None)
    """
    # Use default config if none provided
    if config is None:
        config = Config

    # Set model path to best model if not provided
    if model_path is None:
        model_path = config.MODEL_DIR / 'best_model.pt'

    print("="*70)
    print("Translation Model Evaluation")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Device: {config.DEVICE}")
    print(f"Beam Search: {use_beam_search}")
    if use_beam_search:
        print(f"Beam Width: {config.BEAM_WIDTH}")
    print("="*70)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = Tokenizer(config)
    tokenizer.load(config.MODEL_DIR / 'tokenizer.pkl')

    # Get vocabulary sizes
    src_vocab_size, tgt_vocab_size = tokenizer.get_vocab_sizes()

    # Initialize model
    print("Initializing model...")
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

    # Load checkpoint
    print(f"Loading model from {model_path}...")
    optimizer = torch.optim.Adam(model.parameters())  # Dummy optimizer for loading
    load_checkpoint(model, optimizer, model_path, config.DEVICE)

    # Load test data
    print("\nLoading test data...")
    from dataset import download_tatoeba_dataset, load_data, split_data, TranslationDataset

    file_path = download_tatoeba_dataset(config.DATA_DIR)
    pairs = load_data(file_path, max_length=config.MAX_LENGTH)
    train_pairs, val_pairs, test_pairs = split_data(
        pairs,
        train_ratio=config.TRAIN_SPLIT,
        val_ratio=config.VAL_SPLIT,
        test_ratio=config.TEST_SPLIT
    )

    # Tokenize test data
    test_src, test_tgt = tokenizer.tokenize_pairs(test_pairs)
    test_dataset = TranslationDataset(test_src, test_tgt)

    # Translate test set
    print("\nTranslating test set...")
    predictions, references = translate_dataset(
        model,
        test_dataset,
        tokenizer,
        config.DEVICE,
        config,
        use_beam_search=use_beam_search
    )

    # Calculate BLEU score
    print("\nCalculating BLEU score...")
    bleu_score = calculate_bleu(predictions, references)

    print("\n" + "="*70)
    print(f"BLEU Score: {bleu_score:.2f}")
    print("="*70)

    # Show sample translations
    show_sample_translations(predictions, references, num_samples=10)

    print("\n" + "="*70)
    print("Evaluation completed!")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate translation model')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--greedy', action='store_true', help='Use greedy decoding instead of beam search')

    args = parser.parse_args()

    evaluate(model_path=args.model, use_beam_search=not args.greedy)
