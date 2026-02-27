import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Bidirectional GRU encoder.

    Processes the source sequence in both directions and projects the
    bidirectional hidden states down to `hidden_dim` for the decoder.

    Args:
        vocab_size:  Source vocabulary size.
        embed_dim:   Token embedding dimension.
        hidden_dim:  Per-direction GRU hidden size. Encoder outputs are
                     2 * hidden_dim wide; hidden states are projected back
                     to hidden_dim before being handed to the decoder.
        num_layers:  Number of stacked GRU layers.
        dropout:     Dropout probability (applied to embeddings and between
                     GRU layers when num_layers > 1).
        pad_idx:     Padding token index used to build the source mask.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        # Project concatenated fwd+bwd hidden to hidden_dim for the decoder.
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self, src: torch.Tensor, src_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src:         (batch, src_len) — padded token indices.
            src_lengths: (batch,)         — actual lengths before padding.

        Returns:
            enc_output: (batch, src_len, hidden_dim * 2)
            hidden:     (num_layers, batch, hidden_dim)  — decoder-ready.
        """
        embedded = self.dropout(self.embedding(src))  # (B, T, E)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        enc_output, hidden = self.gru(packed)
        enc_output, _ = nn.utils.rnn.pad_packed_sequence(enc_output, batch_first=True)
        # enc_output: (B, T, H*2)
        # hidden:     (num_layers*2, B, H)

        # Merge fwd/bwd hidden states per layer → (num_layers, B, H).
        hidden = torch.stack(
            [
                torch.tanh(self.fc(torch.cat([hidden[2 * i], hidden[2 * i + 1]], dim=-1)))
                for i in range(self.num_layers)
            ]
        )  # (num_layers, B, H)

        return enc_output, hidden


class Attention(nn.Module):
    """Luong dot-product attention.

    Encoder outputs (width enc_hidden_dim * 2) are projected once to
    dec_hidden_dim, then scores are computed as a plain dot product with
    the decoder's current hidden state — no extra learned parameters beyond
    the projection.

    Args:
        enc_hidden_dim: Per-direction encoder hidden size (encoder outputs
                        are 2 * enc_hidden_dim wide).
        dec_hidden_dim: Decoder hidden size.
    """

    def __init__(self, enc_hidden_dim: int, dec_hidden_dim: int) -> None:
        super().__init__()
        # Align encoder output width to decoder hidden size for the dot product.
        self.enc_proj = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim, bias=False)

    def forward(
        self,
        dec_hidden: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            dec_hidden: (batch, dec_hidden_dim) — current top-layer hidden.
            enc_output: (batch, src_len, enc_hidden_dim * 2).
            src_mask:   (batch, src_len) bool — True for real tokens.

        Returns:
            attn_weights: (batch, src_len) — normalised attention scores.
        """
        enc_proj = self.enc_proj(enc_output)  # (B, T, H_dec)

        # scores = h_t · enc_proj^T  →  (B, 1, H_dec) @ (B, H_dec, T) → (B, T)
        scores = torch.bmm(dec_hidden.unsqueeze(1), enc_proj.transpose(1, 2)).squeeze(1)

        if src_mask is not None:
            scores = scores.masked_fill(~src_mask, float("-inf"))

        return F.softmax(scores, dim=-1)  # (B, T)


class Decoder(nn.Module):
    """GRU decoder with Luong dot-product attention.

    Luong attention is computed with the *current* hidden state (post-GRU),
    so the step order is: embed → GRU → attend → combine → project.

    Args:
        vocab_size:     Target vocabulary size.
        embed_dim:      Token embedding dimension.
        enc_hidden_dim: Per-direction encoder hidden size.
        dec_hidden_dim: Decoder GRU hidden size.
        num_layers:     Number of stacked decoder GRU layers.
        dropout:        Dropout probability.
        pad_idx:        Padding token index.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        enc_hidden_dim: int,
        dec_hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = Attention(enc_hidden_dim, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # GRU input: embedding only (context is applied after the GRU).
        self.gru = nn.GRU(
            embed_dim,
            dec_hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        # Combine layer: tanh(W_c [h_t ; c_t]) → attentional hidden state.
        self.W_c = nn.Linear(dec_hidden_dim + enc_hidden_dim * 2, dec_hidden_dim, bias=False)
        # Final projection from attentional hidden state to vocabulary.
        self.fc_out = nn.Linear(dec_hidden_dim, vocab_size)

    def forward(
        self,
        token: torch.Tensor,
        hidden: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single decoding step.

        Args:
            token:      (batch,)                   — current input token.
            hidden:     (num_layers, batch, H_dec)  — decoder hidden state.
            enc_output: (batch, src_len, H_enc * 2).
            src_mask:   (batch, src_len) bool.

        Returns:
            logits:       (batch, vocab_size).
            hidden:       (num_layers, batch, H_dec) — updated hidden state.
            attn_weights: (batch, src_len).
        """
        embedded = self.dropout(self.embedding(token.unsqueeze(1)))  # (B, 1, E)

        # 1. Run GRU with embedding only.
        gru_out, hidden = self.gru(embedded, hidden)  # (B, 1, H_dec)
        gru_out = gru_out.squeeze(1)                  # (B, H_dec)

        # 2. Attend using the *current* top-layer hidden state.
        attn_weights = self.attention(hidden[-1], enc_output, src_mask)  # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_output).squeeze(1)  # (B, H_enc*2)

        # 3. Combine: h̃_t = tanh(W_c [h_t ; c_t]).
        h_tilde = torch.tanh(self.W_c(torch.cat([gru_out, context], dim=-1)))  # (B, H_dec)

        logits = self.fc_out(self.dropout(h_tilde))  # (B, V)
        return logits, hidden, attn_weights


class Seq2Seq(nn.Module):
    """Sequence-to-sequence model with a BiGRU encoder and GRU decoder.

    The encoder and decoder are built internally from `config`.

    Args:
        pad_idx       (int): Padding token index (from tokenizer).
        sos_idx       (int): Start-of-sequence / BOS token index (from tokenizer).
        eos_idx       (int): End-of-sequence token index (from tokenizer).
        src_vocab_size (int): Source vocabulary size (from tokenizer).
        tgt_vocab_size (int): Target vocabulary size (from tokenizer).
        config        (dict): Architectural and runtime settings (see below).

    Config keys
    -----------
    Encoder architecture:
        enc_embed_dim   (int)
        enc_hidden_dim  (int)   — per-direction hidden size; outputs are 2x wide
        enc_num_layers  (int)
        enc_dropout     (float)

    Decoder architecture:
        dec_embed_dim   (int)
        dec_hidden_dim  (int)
        dec_num_layers  (int)
        dec_dropout     (float)

    Runtime:
        device          (str)   — e.g. "cpu", "cuda", "cuda:1"

    Example::

        config = {
            "enc_embed_dim": 256,    "dec_embed_dim": 256,
            "enc_hidden_dim": 512,   "dec_hidden_dim": 512,
            "enc_num_layers": 2,     "dec_num_layers": 2,
            "enc_dropout": 0.3,      "dec_dropout": 0.3,
            "device": "cuda",
        }
        model = Seq2Seq(
            pad_idx=tokenizer.pad_token_id,
            sos_idx=tokenizer.bos_token_id,
            eos_idx=tokenizer.eos_token_id,
            src_vocab_size=src_tokenizer.vocab_size,
            tgt_vocab_size=tgt_tokenizer.vocab_size,
            config=config,
        )
    """

    def __init__(
        self,
        pad_idx,
        sos_idx,
        eos_idx,
        src_vocab_size,
        tgt_vocab_size,
        config: dict
    ) -> None:
        super().__init__()

        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device  = torch.device(config["device"])

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            embed_dim=config["enc_embed_dim"],
            hidden_dim=config["enc_hidden_dim"],
            num_layers=config["enc_num_layers"],
            dropout=config["enc_dropout"],
            pad_idx=self.pad_idx,
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            embed_dim=config["dec_embed_dim"],
            enc_hidden_dim=config["enc_hidden_dim"],
            dec_hidden_dim=config["dec_hidden_dim"],
            num_layers=config["dec_num_layers"],
            dropout=config["dec_dropout"],
            pad_idx=self.pad_idx,
        )

        self.to(self.device)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
        max_len: int = 100,
    ) -> torch.Tensor:
        """
        Training mode  (tgt is not None):
            Returns output logits of shape (batch, tgt_len, vocab_size).
            Position 0 is unused (corresponds to <sos> input); logits start
            at position 1.

        Inference mode (tgt is None):
            Returns predicted token indices of shape (batch, seq_len),
            including the leading <sos>.  Decoding stops when all sequences
            have emitted <eos> or max_len is reached.

        Args:
            src:                    (batch, src_len).
            src_lengths:            (batch,).
            tgt:                    (batch, tgt_len) or None.
            teacher_forcing_ratio:  Probability of using ground-truth token
                                    as next input during training.
            max_len:                Maximum decoding steps in inference mode.
        """
        batch_size = src.size(0)
        src_mask = src != self.pad_idx  # (B, src_len)

        enc_output, hidden = self.encoder(src, src_lengths)

        if tgt is not None:
            # ----- Training / validation -----
            tgt_len = tgt.size(1)
            vocab_size = self.decoder.fc_out.out_features
            outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=self.device)

            token = tgt[:, 0]  # <sos>
            for t in range(1, tgt_len):
                logits, hidden, _ = self.decoder(token, hidden, enc_output, src_mask)
                outputs[:, t] = logits
                use_teacher = torch.rand(1).item() < teacher_forcing_ratio
                token = tgt[:, t] if use_teacher else logits.argmax(-1)

            return outputs  # (B, tgt_len, V)

        else:
            # ----- Inference (greedy) -----
            tokens = [
                torch.full((batch_size,), self.sos_idx, dtype=torch.long, device=self.device)
            ]
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            for _ in range(max_len):
                logits, hidden, _ = self.decoder(tokens[-1], hidden, enc_output, src_mask)
                token = logits.argmax(-1)  # (B,)
                tokens.append(token)
                finished |= token == self.eos_idx
                if finished.all():
                    break

            return torch.stack(tokens, dim=1)  # (B, seq_len)

    def beam_search(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        beam_width: int = 10,
        max_len: int = 100,
    ) -> torch.Tensor:
        """Beam search decoding.

        Returns the highest-scoring sequence for each item in the batch.
        Output format matches greedy inference from forward(): token indices
        of shape (batch, seq_len) with <sos> at position 0.

        Args:
            src:         (batch, src_len).
            src_lengths: (batch,).
            beam_width:  Number of beams to maintain. Default 10.
            max_len:     Maximum number of decoding steps. Default 100.
        """
        B  = src.size(0)
        K  = beam_width

        src_mask = src != self.pad_idx                    # (B, T_src)
        enc_output, hidden = self.encoder(src, src_lengths)

        T_src = enc_output.size(1)
        H2    = enc_output.size(2)
        NL    = hidden.size(0)
        H     = hidden.size(2)

        # Expand encoder outputs so every beam has its own copy.
        enc_output = (enc_output                          # (B, T, H2)
                      .unsqueeze(1)
                      .expand(B, K, T_src, H2)
                      .reshape(B * K, T_src, H2))
        hidden     = (hidden                              # (NL, B, H)
                      .unsqueeze(2)
                      .expand(NL, B, K, H)
                      .reshape(NL, B * K, H))
        src_mask   = (src_mask                            # (B, T_src)
                      .unsqueeze(1)
                      .expand(B, K, T_src)
                      .reshape(B * K, T_src))

        # scores[b, k]: cumulative log-prob of beam k for batch item b.
        # Only beam 0 is active at the start; the rest are masked to -inf.
        scores = torch.full((B, K), float("-inf"), device=self.device)
        scores[:, 0] = 0.0

        # seqs[b, k]: token sequence for beam k, batch item b.
        seqs = torch.full((B, K, 1), self.sos_idx, dtype=torch.long, device=self.device)

        finished = torch.zeros(B, K, dtype=torch.bool, device=self.device)
        token    = torch.full((B * K,), self.sos_idx, dtype=torch.long, device=self.device)

        batch_idx = torch.arange(B, device=self.device).unsqueeze(1)  # (B, 1)

        for _ in range(max_len):
            logits, hidden, _ = self.decoder(token, hidden, enc_output, src_mask)
            # logits: (B*K, V)

            V        = logits.size(-1)
            log_prob = F.log_softmax(logits, dim=-1).view(B, K, V)  # (B, K, V)

            # Finished beams must stay at eos; force all other tokens to -inf.
            if finished.any():
                eos_only = torch.full((B, K, V), float("-inf"), device=self.device)
                eos_only[:, :, self.eos_idx] = 0.0
                log_prob = torch.where(finished.unsqueeze(-1), eos_only, log_prob)

            # Accumulate scores and find the top-K extensions per batch item.
            total = (scores.unsqueeze(-1) + log_prob).view(B, K * V)   # (B, K*V)
            top_scores, top_idx = total.topk(K, dim=-1)                 # (B, K)

            beam_idx  = top_idx // V   # which beam each winner came from  (B, K)
            token_idx = top_idx  % V   # which token each winner chose     (B, K)

            # Reorder hidden states to match the new beam ordering.
            global_idx = (batch_idx * K + beam_idx).view(-1)            # (B*K,)
            hidden = hidden[:, global_idx, :]                            # (NL, B*K, H)

            # Extend sequences.
            seqs     = seqs[batch_idx, beam_idx]                         # (B, K, t)
            seqs     = torch.cat([seqs, token_idx.unsqueeze(-1)], dim=-1)# (B, K, t+1)

            # Propagate and update finished flags.
            finished = finished[batch_idx, beam_idx] | (token_idx == self.eos_idx)
            scores   = top_scores

            token = token_idx.view(B * K)

            if finished.all():
                break

        # Return the best beam (index 0 after topk) for each batch item.
        return seqs[:, 0, :]  # (B, seq_len) — includes leading <sos>
