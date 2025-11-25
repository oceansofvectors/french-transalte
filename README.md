# English to French Translation Model

A neural machine translation (NMT) model built with PyTorch that translates English sentences to French using a sequence-to-sequence architecture with attention.

## Features

- **Bidirectional LSTM Encoder**: Captures context from both directions
- **LSTM Decoder with Attention**: Focuses on relevant parts of the input
- **Beam Search**: Generates high-quality translations
- **MPS Support**: Optimized for Apple Silicon (M1/M2) Macs
- **30,000 Word Vocabulary**: Large vocabulary for better coverage
- **Checkpointing**: Save and resume training
- **BLEU Score Evaluation**: Measure translation quality
- **Interactive Translation**: Translate sentences interactively

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Source Sentence                      │
│                    (English: "Hello")                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Bidirectional LSTM   │
         │       Encoder          │
         │   (Forward + Backward) │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │    Attention Layer     │
         │  (Focus on relevant    │
         │   source tokens)       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │     LSTM Decoder       │
         │  (Generates target     │
         │   sequence)            │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │    Beam Search         │
         │ (Selects best output)  │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Target Sentence      │
         │  (French: "Bonjour")   │
         └───────────────────────┘
```

## Project Structure

```
translation/
├── data/                    # Dataset storage
├── models/                  # Saved model checkpoints
├── outputs/                 # Training plots and visualizations
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration and hyperparameters
│   ├── tokenizer.py        # Vocabulary and tokenization
│   ├── dataset.py          # Data loading and preprocessing
│   ├── model.py            # Encoder, Decoder, Seq2Seq
│   ├── beam_search.py      # Beam search algorithm
│   ├── train.py            # Training script
│   ├── evaluate.py         # BLEU score evaluation
│   ├── inference.py        # Translation inference
│   └── utils.py            # Helper functions
├── requirements.txt
└── README.md
```

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The model will automatically use:
- **MPS** (Metal Performance Shaders) on Apple Silicon Macs
- **CUDA** on NVIDIA GPUs
- **CPU** as fallback

## Usage

### 1. Training

Train the model on the Tatoeba English-French dataset:

```bash
cd src
python train.py
```

Training will:
- Download the Tatoeba dataset (~175k sentence pairs)
- Build vocabularies for English and French
- Train the model with teacher forcing
- Save checkpoints every epoch
- Use early stopping based on validation loss
- Generate training loss plots

**Training output:**
- `models/best_model.pt` - Best model based on validation loss
- `models/final_model.pt` - Final model after training
- `models/tokenizer.pkl` - Vocabulary and tokenizer
- `outputs/training_loss.png` - Loss curves

**Expected training time:**
- Apple M1/M2 (MPS): ~2-3 hours for 20 epochs
- Modern GPU: ~1-2 hours
- CPU: ~8-12 hours

### 2. Evaluation

Evaluate the model using BLEU score:

```bash
cd src
python evaluate.py
```

Options:
```bash
# Use a specific checkpoint
python evaluate.py --model ../models/checkpoint_epoch_10.pt

# Use greedy decoding instead of beam search
python evaluate.py --greedy
```

### 3. Translation (Inference)

#### Interactive Mode

Translate sentences interactively:

```bash
cd src
python inference.py
```

Example:
```
English to French Translator
======================================================================
Type 'quit' or 'exit' to stop
======================================================================

English: Hello, how are you?
French:  Bonjour, comment allez-vous?

English: I love learning new languages.
French:  J'aime apprendre de nouvelles langues.
```

#### Single Sentence

Translate a single sentence:

```bash
cd src
python inference.py --sentence "Hello, how are you?"
```

#### File Translation

Translate multiple sentences from a file:

```bash
cd src
python inference.py --input input.txt --output output.txt
```

#### Advanced Options

```bash
# Use greedy decoding (faster but less accurate)
python inference.py --sentence "Hello" --greedy

# Use custom beam width
python inference.py --sentence "Hello" --beam-width 10

# Use a specific model checkpoint
python inference.py --sentence "Hello" --model ../models/checkpoint_epoch_15.pt
```

## Configuration

Edit `src/config.py` to customize hyperparameters:

```python
# Model architecture
EMBEDDING_DIM = 256      # Embedding dimension
HIDDEN_DIM = 512         # LSTM hidden dimension
NUM_LAYERS = 2           # Number of LSTM layers
DROPOUT = 0.3            # Dropout rate

# Training
BATCH_SIZE = 64          # Batch size
NUM_EPOCHS = 20          # Number of epochs
LEARNING_RATE = 0.001    # Learning rate
TEACHER_FORCING_RATIO = 0.5  # Teacher forcing ratio

# Beam search
BEAM_WIDTH = 5           # Beam width
LENGTH_PENALTY = 0.6     # Length penalty
```

## Model Details

### Encoder
- Bidirectional LSTM
- Concatenates forward and backward hidden states
- Embedding dimension: 256
- Hidden dimension: 512 (x2 for bidirectional = 1024)
- Number of layers: 2

### Decoder
- Unidirectional LSTM with attention
- Attention mechanism focuses on relevant source tokens
- Output: 30,000-dimensional softmax over vocabulary
- Hidden dimension: 1024

### Training
- Dataset: Tatoeba English-French (~175k pairs)
- Optimizer: Adam
- Loss: Cross-entropy (ignoring padding)
- Gradient clipping: 1.0
- Teacher forcing ratio: 0.5
- Early stopping: Patience of 5 epochs

### Inference
- Beam search with width 5
- Length penalty: 0.6
- Maximum output length: 100 tokens

## Performance

Expected BLEU scores:
- **Beam Search (width=5)**: 25-35 BLEU
- **Greedy Decoding**: 20-30 BLEU

Note: BLEU scores can vary based on training time and hyperparameters.

## Examples

| English | French (Model Output) |
|---------|----------------------|
| Hello | Bonjour |
| How are you? | Comment allez-vous? |
| I love you | Je t'aime |
| Good morning | Bonjour |
| Thank you | Merci |
| Where is the bathroom? | Où sont les toilettes? |

## Troubleshooting

### MPS Issues on Mac

If you encounter MPS errors:
1. Update to the latest macOS version
2. Update PyTorch: `pip install --upgrade torch`
3. If issues persist, the model will fall back to CPU

### Memory Issues

If you run out of memory:
1. Reduce `BATCH_SIZE` in `config.py`
2. Reduce `HIDDEN_DIM` or `EMBEDDING_DIM`
3. Use CPU instead of MPS/CUDA

### Low BLEU Score

If BLEU scores are low:
1. Train for more epochs
2. Increase model capacity (`HIDDEN_DIM`, `NUM_LAYERS`)
3. Adjust `TEACHER_FORCING_RATIO`
4. Increase `BEAM_WIDTH` during inference

## Technical Details

### Mathematical Formulation

The model follows the standard sequence-to-sequence formulation:

**Encoder:**
```
h_t^f, h_t^b = BiLSTM(x_t, h_{t-1}^f, h_{t-1}^b)
h_t = [h_t^f; h_t^b]  # Concatenate forward and backward
```

**Attention:**
```
e_tj = v^T tanh(W[s_{t-1}; h_j])
α_tj = softmax(e_tj)
c_t = Σ α_tj h_j
```

**Decoder:**
```
s_t = LSTM([y_{t-1}; c_t], s_{t-1})
p(y_t | y_1, ..., y_{t-1}, x) = softmax(W[s_t; c_t; y_{t-1}])
```

### Beam Search

Beam search explores multiple hypotheses in parallel, keeping the top-k most likely sequences at each step, where k is the beam width.

## License

This project is for educational purposes.

## Dataset

This project uses the Tatoeba English-French dataset from [manythings.org](https://www.manythings.org/anki/).

## Future Improvements

- [ ] Add transformer architecture option
- [ ] Support for more language pairs
- [ ] Subword tokenization (BPE/WordPiece)
- [ ] Attention visualization
- [ ] Pre-trained embeddings (Word2Vec, GloVe)
- [ ] Multi-head attention
