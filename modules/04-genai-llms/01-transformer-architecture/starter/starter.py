# Dependencies: torch>=2.1, numpy>=1.24, tiktoken>=0.5
# Node: 4.1 — Transformer Architecture — Encoder-Decoder
# Run: python starter.py

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    """Hyperparameters for minGPT. Small enough to train on CPU."""
    vocab_size: int = 65        # updated at runtime after building char vocab
    block_size: int = 64        # context window length (number of tokens)
    n_layer: int = 4            # number of stacked transformer blocks
    n_head: int = 4             # number of attention heads
    n_embd: int = 128           # embedding / residual stream dimension
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ── Tiny Dataset ──────────────────────────────────────────────────────────────

TINY_SHAKESPEARE = """\
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
"""


# ── Fully Implemented Helpers ─────────────────────────────────────────────────

def load_data(cfg: GPTConfig) -> Tuple[torch.Tensor, torch.Tensor, dict, dict]:
    """
    Build a character-level vocabulary from TINY_SHAKESPEARE.
    Updates cfg.vocab_size in-place.
    Returns train tensor, val tensor, stoi dict, itos dict.
    """
    text = TINY_SHAKESPEARE
    chars = sorted(set(text))
    cfg.vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    return data[:split], data[split:], stoi, itos


def get_batch(
    data: torch.Tensor,
    cfg: GPTConfig,
    batch_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch of (input, target) token sequences.
    target[i] is input[i] shifted one position to the right — the next-token prediction label.
    """
    ix = torch.randint(len(data) - cfg.block_size, (batch_size,))
    x = torch.stack([data[i : i + cfg.block_size] for i in ix]).to(cfg.device)
    y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in ix]).to(cfg.device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    cfg: GPTConfig,
    eval_iters: int = 20,
) -> dict:
    """
    Evaluate average cross-entropy loss over eval_iters batches on both splits.
    Switches model to eval mode temporarily (disables dropout).
    """
    model.eval()
    results = {}
    for split_name, data in [("train", train_data), ("val", val_data)]:
        losses = [
            model(x, y)[1].item()
            for x, y in (get_batch(data, cfg) for _ in range(eval_iters))
        ]
        results[split_name] = float(np.mean(losses))
    model.train()
    return results


# ── Student Implementations ───────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    The causal mask ensures token i can only attend to tokens 0 … i,
    mirroring how a REST streaming API only has access to bytes already sent.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"

        # Single linear layer that projects x → Q, K, V concatenated
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        # Output projection back to residual stream dimension
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd

        # Lower-triangular causal mask: mask[i, j] = 1 if j <= i, else 0
        # Registered as a buffer so it moves with the model to CUDA automatically
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)  — batch, sequence length, embedding dim

        Returns:
            out: (B, T, C)

        Implementation steps:
          1. Project x through self.c_attn → (B, T, 3*C).
             Split along last dim into Q, K, V each of shape (B, T, C).
          2. Reshape Q, K, V to (B, n_head, T, head_dim)
             where head_dim = n_embd // n_head.
          3. scores = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)  → (B, n_head, T, T)
          4. Apply causal mask: scores = scores.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
          5. attn_weights = self.attn_drop(F.softmax(scores, dim=-1))
          6. out = attn_weights @ V                                   → (B, n_head, T, head_dim)
          7. Transpose and reshape back to (B, T, C).
          8. Return self.resid_drop(self.c_proj(out))
        """
        raise NotImplementedError("TODO: implement this")


class TransformerBlock(nn.Module):
    """
    Single GPT transformer block:
        x → LayerNorm → CausalSelfAttention → residual →
            LayerNorm → MLP (GELU) → residual
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        # The MLP expands to 4× width, applies GELU, then projects back
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)

        Returns:
            x: (B, T, C) — same shape, residual stream updated

        Implementation steps (2 lines):
          1. x = x + self.attn(self.ln1(x))   ← attention sub-layer with residual
          2. x = x + self.mlp(self.ln2(x))    ← feed-forward sub-layer with residual
        """
        raise NotImplementedError("TODO: implement this")


class MiniGPT(nn.Module):
    """
    Decoder-only GPT for character-level language modelling.
    Identical in structure to GPT-2 (small) — just much smaller dimensions.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Token embedding table: maps each token id → a learnable vector in R^n_embd
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # Position embedding table: maps each position 0…block_size-1 → R^n_embd
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        # Final layer norm before the language model head
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        # Linear projection from residual stream → vocabulary logits (no bias, per GPT-2)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx:     (B, T) integer token indices in [0, vocab_size)
            targets: (B, T) next-token ground-truth indices (optional)

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy if targets provided, else None

        Implementation steps:
          1. B, T = idx.shape
          2. tok_emb = self.token_emb(idx)                       → (B, T, C)
          3. pos = torch.arange(T, device=idx.device)
             pos_emb = self.pos_emb(pos)                         → (T, C), broadcasts over B
          4. x = self.drop(tok_emb + pos_emb)
          5. x = self.blocks(x)
          6. x = self.ln_f(x)
          7. logits = self.head(x)                               → (B, T, vocab_size)
          8. If targets is not None:
               loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
             else:
               loss = None
          9. return logits, loss
        """
        raise NotImplementedError("TODO: implement this")

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressively sample max_new_tokens tokens, appending each to idx.

        Args:
            idx:            (1, T) seed token indices
            max_new_tokens: how many new tokens to produce
            temperature:    divide logits before softmax;
                            <1.0 sharpens distribution (more repetitive),
                            >1.0 flattens it (more random)

        Returns:
            idx: (1, T + max_new_tokens)

        Implementation steps (repeat max_new_tokens times):
          1. idx_cond = idx[:, -self.cfg.block_size:]   ← crop to context window
          2. logits, _ = self(idx_cond)                 ← forward pass, ignore loss
          3. logits = logits[:, -1, :] / temperature    ← logits at final time step
          4. probs = F.softmax(logits, dim=-1)
          5. next_tok = torch.multinomial(probs, num_samples=1)
          6. idx = torch.cat([idx, next_tok], dim=1)
        """
        raise NotImplementedError("TODO: implement this")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(42)
    cfg = GPTConfig()

    # ── Task 5: Load data ────────────────────────────────────────────────────
    train_data, val_data, stoi, itos = load_data(cfg)
    print(f"Vocab size : {cfg.vocab_size} characters")
    print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

    # ── Build model (requires Tasks 1-3 to be implemented) ───────────────────
    model = MiniGPT(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # ── Task 5: Training loop — 500 steps ────────────────────────────────────
    print("\nTraining …")
    for step in range(500):
        x, y = get_batch(train_data, cfg)
        _, loss = model(x, y)         # MiniGPT.forward() — Task 3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            metrics = estimate_loss(model, train_data, val_data, cfg)
            print(
                f"  step {step:>4d} | "
                f"train loss: {metrics['train']:.4f} | "
                f"val loss: {metrics['val']:.4f}"
            )

    # ── Task 6: Text generation ───────────────────────────────────────────────
    print("\nGenerating text …")
    seed_text = "First Citizen:"
    seed_ids = torch.tensor(
        [[stoi.get(c, 0) for c in seed_text]], dtype=torch.long, device=cfg.device
    )
    generated_ids = model.generate(seed_ids, max_new_tokens=200, temperature=0.8)  # Task 4
    generated_text = "".join(itos[i] for i in generated_ids[0].tolist())
    print(f"\n--- Generated Text ---\n{generated_text}\n")


if __name__ == "__main__":
    main()
