"""
Neural machine translation model with encoder-decoder architecture.
Encoder: Bidirectional LSTM
Decoder: LSTM with attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    """Bidirectional LSTM Encoder."""

    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, dropout):
        """
        Initialize the encoder.

        Args:
            input_dim: Size of source vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        """
        Forward pass.

        Args:
            src: Source sequences [batch_size, src_len]
            src_lengths: Actual lengths of source sequences [batch_size]

        Returns:
            outputs: Encoder outputs [batch_size, src_len, hidden_dim * 2]
            hidden: Final hidden state [num_layers, batch_size, hidden_dim * 2]
            cell: Final cell state [num_layers, batch_size, hidden_dim * 2]
        """
        # Embed source tokens
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, embedding_dim]

        # Pack padded sequences
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Forward pass through LSTM
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)

        # Unpack sequences
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: [batch_size, src_len, hidden_dim * 2]

        # hidden and cell: [num_layers * 2, batch_size, hidden_dim]
        # We need to concatenate forward and backward hidden states
        # Reshape to [num_layers, 2, batch_size, hidden_dim]
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)

        # Concatenate forward and backward
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
        # hidden and cell: [num_layers, batch_size, hidden_dim * 2]

        return outputs, hidden, cell


class Attention(nn.Module):
    """Attention mechanism for the decoder."""

    def __init__(self, hidden_dim):
        """
        Initialize attention.

        Args:
            hidden_dim: Hidden state dimension (encoder outputs are hidden_dim * 2)
        """
        super().__init__()

        # Attention weights
        # Concatenating hidden (hidden_dim * 2) + encoder_outputs (hidden_dim * 2) = hidden_dim * 4
        self.attn = nn.Linear(hidden_dim * 4, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        """
        Compute attention weights and context vector.

        Args:
            hidden: Current decoder hidden state [batch_size, hidden_dim * 2]
            encoder_outputs: Encoder outputs [batch_size, src_len, hidden_dim * 2]
            mask: Mask for padding [batch_size, src_len]

        Returns:
            context: Context vector [batch_size, hidden_dim * 2]
            attn_weights: Attention weights [batch_size, src_len]
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden: [batch_size, src_len, hidden_dim * 2]

        # Concatenate and compute energy
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        # energy: [batch_size, src_len, hidden_dim]

        # Compute attention scores
        attention = self.v(energy).squeeze(2)
        # attention: [batch_size, src_len]

        # Mask padding tokens
        attention = attention.masked_fill(mask == 0, -1e10)

        # Compute attention weights
        attn_weights = F.softmax(attention, dim=1)
        # attn_weights: [batch_size, src_len]

        # Compute context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        # context: [batch_size, 1, hidden_dim * 2]

        context = context.squeeze(1)
        # context: [batch_size, hidden_dim * 2]

        return context, attn_weights


class Decoder(nn.Module):
    """LSTM Decoder with attention."""

    def __init__(self, output_dim, embedding_dim, hidden_dim, num_layers, dropout):
        """
        Initialize the decoder.

        Args:
            output_dim: Size of target vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: Hidden state dimension (encoder outputs are hidden_dim * 2)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(output_dim, embedding_dim)

        # LSTM (not bidirectional)
        # Input is embedding + context vector
        self.lstm = nn.LSTM(
            embedding_dim + hidden_dim * 2,
            hidden_dim * 2,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Attention
        self.attention = Attention(hidden_dim)

        # Output layer
        self.fc_out = nn.Linear(hidden_dim * 4 + embedding_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs, mask):
        """
        Forward pass for one time step.

        Args:
            input_token: Current input token [batch_size]
            hidden: Previous hidden state [num_layers, batch_size, hidden_dim * 2]
            cell: Previous cell state [num_layers, batch_size, hidden_dim * 2]
            encoder_outputs: Encoder outputs [batch_size, src_len, hidden_dim * 2]
            mask: Mask for padding [batch_size, src_len]

        Returns:
            prediction: Output predictions [batch_size, output_dim]
            hidden: Updated hidden state [num_layers, batch_size, hidden_dim * 2]
            cell: Updated cell state [num_layers, batch_size, hidden_dim * 2]
            attn_weights: Attention weights [batch_size, src_len]
        """
        # Embed input token
        input_token = input_token.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input_token))  # [batch_size, 1, embedding_dim]

        # Compute attention using the last layer's hidden state
        context, attn_weights = self.attention(hidden[-1], encoder_outputs, mask)
        # context: [batch_size, hidden_dim * 2]

        # Concatenate embedded input and context
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        # lstm_input: [batch_size, 1, embedding_dim + hidden_dim * 2]

        # LSTM forward pass
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: [batch_size, 1, hidden_dim * 2]

        # Concatenate output, context, and embedded input for prediction
        output = output.squeeze(1)
        context = context
        embedded = embedded.squeeze(1)

        prediction = self.fc_out(torch.cat([output, context, embedded], dim=1))
        # prediction: [batch_size, output_dim]

        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    """Sequence-to-sequence model."""

    def __init__(self, encoder, decoder, device):
        """
        Initialize the seq2seq model.

        Args:
            encoder: Encoder module
            decoder: Decoder module
            device: Device to run on
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, src):
        """
        Create mask for source sequences.

        Args:
            src: Source sequences [batch_size, src_len]

        Returns:
            mask: Mask tensor [batch_size, src_len]
        """
        # Assuming 0 is the padding index
        mask = (src != 0).to(self.device)
        return mask

    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        """
        Forward pass.

        Args:
            src: Source sequences [batch_size, src_len]
            src_lengths: Actual lengths of source sequences [batch_size]
            tgt: Target sequences [batch_size, tgt_len]
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            outputs: Decoder outputs [batch_size, tgt_len, output_dim]
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_dim

        # Store outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # Encode source
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

        # Create mask for attention
        mask = self.create_mask(src)

        # First input to decoder is <sos> token
        input_token = tgt[:, 0]

        # Decode
        for t in range(1, tgt_len):
            # Forward pass through decoder
            output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs, mask)

            # Store output
            outputs[:, t, :] = output

            # Decide if we use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio

            # Get predicted token
            top1 = output.argmax(1)

            # Next input is either target token or predicted token
            input_token = tgt[:, t] if teacher_force else top1

        return outputs
