# Assignment 4.1 — Construct minGPT from Scratch

## Objective

Build a decoder-only GPT model from raw PyTorch primitives — implementing causal self-attention, transformer blocks, and autoregressive generation — to prove you understand every layer before you ever call `from transformers import GPT2`.

## Background

The lesson established that GPT is a stack of transformer blocks, each combining masked multi-head self-attention with a position-wise feed-forward network and residual connections. Refer back to the Theory section of `lesson.md` for the scaled dot-product attention formula and the residual block diagram. This assignment has you wire those primitives together from scratch using a 128-dimensional, 4-layer, 4-head model trained on a character-level corpus — small enough to run on a CPU in under 10 minutes. Each task maps directly to one equation from the lesson: implement it, watch the loss fall, then generate text.

## Setup

```bash
pip install torch>=2.1 numpy>=1.24 tiktoken>=0.5
```

## Tasks

1. **Implement `CausalSelfAttention.forward()`** — Project input `x` into Q, K, V via `self.c_attn`; reshape into `n_head` heads; compute scaled dot-product attention scores (`Q @ K.T / sqrt(head_dim)`); mask future positions to `-inf` using `self.mask`; apply softmax + `self.attn_drop`; compute the weighted sum over V; reshape back and apply `self.c_proj` + `self.resid_drop`. Verify that `attn_weights[:, :, 0, 1]` is zero for all batches and heads after masking.

2. **Implement `TransformerBlock.forward()`** — Two lines: apply pre-norm attention with a residual (`x = x + self.attn(self.ln1(x))`), then apply pre-norm MLP with a residual (`x = x + self.mlp(self.ln2(x))`). Run a single forward pass with a random `(2, 16, 128)` tensor to confirm the output shape is unchanged.

3. **Implement `MiniGPT.forward()`** — Sum token embeddings and position embeddings; pass through `self.drop`, `self.blocks`, and `self.ln_f`; project to `vocab_size` logits. When `targets` is provided, compute `F.cross_entropy` over the flattened `(B*T, vocab_size)` logits and `(B*T,)` targets and return both logits and the scalar loss.

4. **Implement `MiniGPT.generate()`** — Run the autoregressive loop `max_new_tokens` times: crop the running context to the last `block_size` tokens, forward-pass to get logits, take the logit slice at the final time step, divide by `temperature`, apply softmax, sample one token with `torch.multinomial`, and append it to `idx`.

5. **Train the model** — Run `main()` for 500 optimiser steps on the included `TINY_SHAKESPEARE` corpus. Confirm that training loss falls below **1.5** by step 500.

6. **Generate text** — After training, observe the output of `model.generate()` called with seed `"First Citizen:"` and `max_new_tokens=200`. The generated text must contain recognisable English words — not random noise.

## Expected Output

```
Vocab size : 62 characters
Train tokens: 1,189  |  Val tokens: 133
Model parameters: 793,410

Training …
  step    0 | train loss: 4.1xxx | val loss: 4.1xxx
  step  100 | train loss: 2.3xxx | val loss: 2.4xxx
  step  200 | train loss: 1.9xxx | val loss: 2.0xxx
  step  300 | train loss: 1.7xxx | val loss: 1.8xxx
  step  400 | train loss: 1.5xxx | val loss: 1.6xxx

Generating text …

--- Generated Text ---
First Citizen:
We are accounted poor citizens, the patricians good.
[...~200 chars of plausible Shakespeare-style character sequences...]
```

> Training loss at step 400 should be in the range **1.4 – 1.8**. Exact values vary by run due to random batch sampling. Exact parameter count may differ slightly if you modify `GPTConfig`.

## Evaluation Criteria

- [ ] `CausalSelfAttention.forward()` is implemented and `attn_weights[:, :, 0, 1]` is zero after masking (token 0 cannot attend to token 1).
- [ ] `TransformerBlock.forward()` uses pre-norm (`LayerNorm` applied before the sub-layer, not after) on both the attention and MLP paths.
- [ ] `MiniGPT.forward()` returns logits of shape `(batch, seq_len, vocab_size)` and a scalar cross-entropy loss when `targets` is provided.
- [ ] `MiniGPT.generate()` returns a tensor of shape `(1, T + max_new_tokens)` and does not raise an error for any `temperature > 0`.
- [ ] Training loss is below **1.5** at step 500 with the default `GPTConfig`.
- [ ] Generated text from the `"First Citizen:"` seed contains recognisable English words (not pure character noise).

## Extension Challenge

Replace the learned absolute position embeddings (`nn.Embedding`) with **Rotary Position Embeddings (RoPE)** as used in Llama-2/3. Implement the complex-plane rotation inside `CausalSelfAttention.forward()` so that relative position information is encoded directly in the Q and K projections rather than added to the input. Verify the model still converges to loss < 1.5 on the same dataset. No starter code provided — reference Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position Embedding" or the Llama source in `meta-llama/llama`.
