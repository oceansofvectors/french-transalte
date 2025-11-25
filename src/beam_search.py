"""
Beam search implementation for translation inference.
"""

import torch
import torch.nn.functional as F


class BeamSearchNode:
    """Node in the beam search tree."""

    def __init__(self, hidden, cell, prev_node, token_idx, log_prob, length):
        """
        Initialize a beam search node.

        Args:
            hidden: Hidden state
            cell: Cell state
            prev_node: Previous node in the sequence
            token_idx: Token index for this node
            log_prob: Cumulative log probability
            length: Current sequence length
        """
        self.hidden = hidden
        self.cell = cell
        self.prev_node = prev_node
        self.token_idx = token_idx
        self.log_prob = log_prob
        self.length = length

    def eval(self, length_penalty=0.6):
        """
        Evaluate the node score with length normalization.

        Args:
            length_penalty: Length penalty factor (0.0-1.0)

        Returns:
            Normalized score
        """
        # Length normalization to avoid bias towards shorter sequences
        return self.log_prob / (self.length ** length_penalty)


def beam_search(model, src, max_length, beam_width, device, sos_idx=1, eos_idx=2, length_penalty=0.6):
    """
    Perform beam search for translation.

    Args:
        model: Trained seq2seq model
        src: Source sequence [1, src_len]
        max_length: Maximum output length
        beam_width: Width of the beam
        device: Device to run on
        sos_idx: Start-of-sequence token index
        eos_idx: End-of-sequence token index
        length_penalty: Length penalty factor

    Returns:
        best_sequence: List of token indices for the best translation
    """
    model.eval()

    with torch.no_grad():
        # Encode source
        src_lengths = torch.tensor([src.shape[1]], dtype=torch.long).to(device)
        encoder_outputs, hidden, cell = model.encoder(src, src_lengths)

        # Create mask
        mask = model.create_mask(src)

        # Initialize with start token
        start_token = torch.tensor([sos_idx], dtype=torch.long).to(device)

        # Initialize beam with start node
        start_node = BeamSearchNode(
            hidden=hidden,
            cell=cell,
            prev_node=None,
            token_idx=sos_idx,
            log_prob=0.0,
            length=1
        )

        # List of nodes to expand
        nodes = [start_node]
        # List of completed sequences
        finished_nodes = []

        # Beam search
        for step in range(max_length):
            if len(nodes) == 0:
                break

            # Store all candidates from current nodes
            all_candidates = []

            for node in nodes:
                # If we've reached EOS, add to finished nodes
                if node.token_idx == eos_idx:
                    finished_nodes.append(node)
                    continue

                # Get decoder output
                input_token = torch.tensor([node.token_idx], dtype=torch.long).to(device)
                output, hidden, cell, _ = model.decoder(
                    input_token,
                    node.hidden,
                    node.cell,
                    encoder_outputs,
                    mask
                )

                # Get log probabilities
                log_probs = F.log_softmax(output, dim=1)

                # Get top k candidates
                top_log_probs, top_indices = torch.topk(log_probs, beam_width)

                # Create new nodes for each candidate
                for k in range(beam_width):
                    token_idx = top_indices[0, k].item()
                    log_prob = top_log_probs[0, k].item()

                    new_node = BeamSearchNode(
                        hidden=hidden,
                        cell=cell,
                        prev_node=node,
                        token_idx=token_idx,
                        log_prob=node.log_prob + log_prob,
                        length=node.length + 1
                    )

                    all_candidates.append(new_node)

            # Select top beam_width nodes
            if len(all_candidates) > 0:
                nodes = sorted(all_candidates, key=lambda x: x.eval(length_penalty), reverse=True)[:beam_width]

        # If no sequences finished, use the best partial sequence
        if len(finished_nodes) == 0:
            finished_nodes = nodes

        # Select best sequence
        best_node = max(finished_nodes, key=lambda x: x.eval(length_penalty))

        # Reconstruct sequence
        sequence = []
        current_node = best_node

        while current_node is not None:
            sequence.append(current_node.token_idx)
            current_node = current_node.prev_node

        # Reverse to get correct order (excluding SOS token)
        sequence = sequence[::-1][1:]

        return sequence


def greedy_decode(model, src, max_length, device, sos_idx=1, eos_idx=2):
    """
    Perform greedy decoding for translation (faster but less accurate than beam search).

    Args:
        model: Trained seq2seq model
        src: Source sequence [1, src_len]
        max_length: Maximum output length
        device: Device to run on
        sos_idx: Start-of-sequence token index
        eos_idx: End-of-sequence token index

    Returns:
        sequence: List of token indices for the translation
    """
    model.eval()

    with torch.no_grad():
        # Encode source
        src_lengths = torch.tensor([src.shape[1]], dtype=torch.long).to(device)
        encoder_outputs, hidden, cell = model.encoder(src, src_lengths)

        # Create mask
        mask = model.create_mask(src)

        # Initialize with start token
        input_token = torch.tensor([sos_idx], dtype=torch.long).to(device)

        # Store output sequence
        sequence = []

        # Decode
        for t in range(max_length):
            # Forward pass
            output, hidden, cell, _ = model.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs,
                mask
            )

            # Get most likely token
            predicted_token = output.argmax(1).item()

            # Add to sequence
            sequence.append(predicted_token)

            # Stop if we predict EOS
            if predicted_token == eos_idx:
                break

            # Next input
            input_token = torch.tensor([predicted_token], dtype=torch.long).to(device)

        return sequence
