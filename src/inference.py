"""
Inference script for translating English sentences to French.
"""

import torch
import sys

from config import Config
from tokenizer import Tokenizer
from model import Encoder, Decoder, Seq2Seq
from beam_search import beam_search, greedy_decode
from utils import load_checkpoint


class Translator:
    """Translator class for easy inference."""

    def __init__(self, model_path=None, config=None):
        """
        Initialize the translator.

        Args:
            model_path: Path to model checkpoint (uses best model if None)
            config: Configuration object (uses default if None)
        """
        # Use default config if none provided
        if config is None:
            config = Config

        self.config = config

        # Set model path to best model if not provided
        if model_path is None:
            model_path = config.MODEL_DIR / 'best_model.pt'

        print("Loading translator...")
        print(f"Device: {config.DEVICE}")

        # Load tokenizer
        self.tokenizer = Tokenizer(config)
        tokenizer_path = config.MODEL_DIR / 'tokenizer.pkl'

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                "Please train the model first using train.py"
            )

        self.tokenizer.load(tokenizer_path)

        # Get vocabulary sizes
        src_vocab_size, tgt_vocab_size = self.tokenizer.get_vocab_sizes()

        # Initialize model
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

        self.model = Seq2Seq(encoder, decoder, config.DEVICE).to(config.DEVICE)

        # Load checkpoint
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please train the model first using train.py"
            )

        optimizer = torch.optim.Adam(self.model.parameters())
        load_checkpoint(self.model, optimizer, model_path, config.DEVICE)

        self.model.eval()
        print("Translator loaded successfully!")

    def translate(self, sentence, use_beam_search=True, beam_width=None):
        """
        Translate an English sentence to French.

        Args:
            sentence: English sentence to translate
            use_beam_search: Whether to use beam search (vs greedy)
            beam_width: Beam width (uses config default if None)

        Returns:
            Translated French sentence
        """
        if beam_width is None:
            beam_width = self.config.BEAM_WIDTH

        # Tokenize input
        src_tokens = self.tokenizer.encode_sentence(sentence, is_target=False)
        src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(self.config.DEVICE)

        # Translate
        with torch.no_grad():
            if use_beam_search:
                predicted_tokens = beam_search(
                    self.model,
                    src_tensor,
                    max_length=self.config.MAX_LENGTH,
                    beam_width=beam_width,
                    device=self.config.DEVICE,
                    sos_idx=self.config.SOS_IDX,
                    eos_idx=self.config.EOS_IDX,
                    length_penalty=self.config.LENGTH_PENALTY
                )
            else:
                predicted_tokens = greedy_decode(
                    self.model,
                    src_tensor,
                    max_length=self.config.MAX_LENGTH,
                    device=self.config.DEVICE,
                    sos_idx=self.config.SOS_IDX,
                    eos_idx=self.config.EOS_IDX
                )

        # Decode
        translation = self.tokenizer.decode_sentence(predicted_tokens, is_target=True)
        return translation


def interactive_mode():
    """Run translator in interactive mode."""
    print("="*70)
    print("English to French Translator")
    print("="*70)
    print("Type 'quit' or 'exit' to stop")
    print("="*70 + "\n")

    try:
        translator = Translator()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    while True:
        try:
            # Get input
            sentence = input("\nEnglish: ").strip()

            # Check for exit
            if sentence.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            # Skip empty input
            if not sentence:
                continue

            # Translate
            translation = translator.translate(sentence)

            # Display result
            print(f"French:  {translation}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def translate_file(input_file, output_file, model_path=None):
    """
    Translate sentences from a file.

    Args:
        input_file: Path to input file (one sentence per line)
        output_file: Path to output file
        model_path: Path to model checkpoint
    """
    print(f"Translating from {input_file} to {output_file}...")

    try:
        translator = Translator(model_path=model_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    # Translate
    translations = []
    for i, sentence in enumerate(sentences, 1):
        print(f"Translating {i}/{len(sentences)}...", end='\r')
        translation = translator.translate(sentence)
        translations.append(translation)

    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for translation in translations:
            f.write(translation + '\n')

    print(f"\nTranslation complete! Output saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Translate English to French')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--sentence', type=str, default=None, help='Sentence to translate')
    parser.add_argument('--input', type=str, default=None, help='Input file (one sentence per line)')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    parser.add_argument('--greedy', action='store_true', help='Use greedy decoding')
    parser.add_argument('--beam-width', type=int, default=None, help='Beam width')

    args = parser.parse_args()

    if args.input and args.output:
        # File translation mode
        translate_file(args.input, args.output, args.model)
    elif args.sentence:
        # Single sentence mode
        try:
            translator = Translator(model_path=args.model)
            translation = translator.translate(
                args.sentence,
                use_beam_search=not args.greedy,
                beam_width=args.beam_width
            )
            print(f"\nEnglish: {args.sentence}")
            print(f"French:  {translation}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
    else:
        # Interactive mode
        interactive_mode()
