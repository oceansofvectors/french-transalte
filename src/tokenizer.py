"""
Tokenizer and vocabulary builder for the translation model.
"""

from collections import Counter
from tqdm import tqdm
import pickle


class Tokenizer:
    """Tokenizer for building vocabulary and converting between tokens and indices."""

    def __init__(self, config):
        """
        Initialize the tokenizer.

        Args:
            config: Configuration object
        """
        self.config = config

        # Special tokens
        self.pad_token = config.PAD_TOKEN
        self.sos_token = config.SOS_TOKEN
        self.eos_token = config.EOS_TOKEN
        self.unk_token = config.UNK_TOKEN

        self.pad_idx = config.PAD_IDX
        self.sos_idx = config.SOS_IDX
        self.eos_idx = config.EOS_IDX
        self.unk_idx = config.UNK_IDX

        # Vocabularies
        self.src_word2idx = {}
        self.src_idx2word = {}
        self.tgt_word2idx = {}
        self.tgt_idx2word = {}

    def build_vocab(self, pairs):
        """
        Build vocabularies from sentence pairs.

        Args:
            pairs: List of (source, target) sentence pairs
        """
        print("Building vocabulary...")

        # Count word frequencies
        src_counter = Counter()
        tgt_counter = Counter()

        for src, tgt in tqdm(pairs, desc="Counting words"):
            src_counter.update(src.lower().split())
            tgt_counter.update(tgt.lower().split())

        # Build source vocabulary
        self.src_word2idx = {
            self.pad_token: self.pad_idx,
            self.sos_token: self.sos_idx,
            self.eos_token: self.eos_idx,
            self.unk_token: self.unk_idx,
        }

        # Add most common words (up to vocab size)
        idx = len(self.src_word2idx)
        for word, freq in src_counter.most_common(self.config.VOCAB_SIZE - 4):
            if freq >= self.config.MIN_FREQ:
                self.src_word2idx[word] = idx
                idx += 1

        self.src_idx2word = {idx: word for word, idx in self.src_word2idx.items()}

        # Build target vocabulary
        self.tgt_word2idx = {
            self.pad_token: self.pad_idx,
            self.sos_token: self.sos_idx,
            self.eos_token: self.eos_idx,
            self.unk_token: self.unk_idx,
        }

        # Add most common words (up to vocab size)
        idx = len(self.tgt_word2idx)
        for word, freq in tgt_counter.most_common(self.config.VOCAB_SIZE - 4):
            if freq >= self.config.MIN_FREQ:
                self.tgt_word2idx[word] = idx
                idx += 1

        self.tgt_idx2word = {idx: word for word, idx in self.tgt_word2idx.items()}

        print(f"Source vocabulary size: {len(self.src_word2idx)}")
        print(f"Target vocabulary size: {len(self.tgt_word2idx)}")

    def encode_sentence(self, sentence, is_target=False):
        """
        Convert a sentence to a list of token indices.

        Args:
            sentence: Input sentence string
            is_target: Whether this is a target sentence (adds SOS/EOS tokens)

        Returns:
            List of token indices
        """
        word2idx = self.tgt_word2idx if is_target else self.src_word2idx
        tokens = sentence.lower().split()

        indices = [word2idx.get(token, self.unk_idx) for token in tokens]

        # Add SOS and EOS for target sentences
        if is_target:
            indices = [self.sos_idx] + indices + [self.eos_idx]

        return indices

    def decode_sentence(self, indices, is_target=False):
        """
        Convert a list of token indices to a sentence.

        Args:
            indices: List of token indices
            is_target: Whether this is a target sentence

        Returns:
            Sentence string
        """
        idx2word = self.tgt_idx2word if is_target else self.src_idx2word

        words = []
        for idx in indices:
            # Stop at EOS token
            if idx == self.eos_idx:
                break
            # Skip PAD and SOS tokens
            if idx in [self.pad_idx, self.sos_idx]:
                continue

            word = idx2word.get(idx, self.unk_token)
            words.append(word)

        return ' '.join(words)

    def decode(self, indices, is_target=False):
        """Alias for decode_sentence."""
        return self.decode_sentence(indices, is_target)

    def tokenize_pairs(self, pairs):
        """
        Tokenize a list of sentence pairs.

        Args:
            pairs: List of (source, target) sentence pairs

        Returns:
            src_indices: List of source token index lists
            tgt_indices: List of target token index lists
        """
        src_indices = []
        tgt_indices = []

        for src, tgt in tqdm(pairs, desc="Tokenizing"):
            src_indices.append(self.encode_sentence(src, is_target=False))
            tgt_indices.append(self.encode_sentence(tgt, is_target=True))

        return src_indices, tgt_indices

    def save(self, path):
        """
        Save the tokenizer to disk.

        Args:
            path: Path to save the tokenizer
        """
        data = {
            'src_word2idx': self.src_word2idx,
            'src_idx2word': self.src_idx2word,
            'tgt_word2idx': self.tgt_word2idx,
            'tgt_idx2word': self.tgt_idx2word,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Tokenizer saved to {path}")

    def load(self, path):
        """
        Load the tokenizer from disk.

        Args:
            path: Path to load the tokenizer from
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.src_word2idx = data['src_word2idx']
        self.src_idx2word = data['src_idx2word']
        self.tgt_word2idx = data['tgt_word2idx']
        self.tgt_idx2word = data['tgt_idx2word']
        print(f"Tokenizer loaded from {path}")

    def get_vocab_sizes(self):
        """
        Get vocabulary sizes.

        Returns:
            Tuple of (source_vocab_size, target_vocab_size)
        """
        return len(self.src_word2idx), len(self.tgt_word2idx)
