# Dependencies: torch>=2.1, transformers>=4.40, peft>=0.10, datasets>=2.18, bitsandbytes>=0.43, trl>=0.8
# Node: 4.2 — Fine-Tuning: LoRA & QLoRA
# Run: python starter.py
#
# Model access: Meta Llama-3-8B is gated.
#   1. Accept terms at: https://huggingface.co/meta-llama/Meta-Llama-3-8B
#   2. Run: huggingface-cli login
#
# CPU / low-VRAM fallback (no token required, ~20 min on CPU):
#   Change MODEL_ID below to "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

import os
import torch
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from typing import Any

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_ID: str = os.environ.get("MODEL_ID", "meta-llama/Meta-Llama-3-8B")
# Fallback: MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DATASET_NAME: str = "tatsu-lab/alpaca"
MAX_SAMPLES: int = 1000      # small subset — full dataset is 52k records
MAX_SEQ_LEN: int = 512
LORA_RANK: int = 16
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.05
TRAINING_STEPS: int = 50
OUTPUT_DIR: str = "./lora-output"
# ───────────────────────────────────────────────────────────────────────────────


# ── Fully implemented helpers ──────────────────────────────────────────────────

def count_trainable_parameters(model: Any) -> None:
    """Print trainable vs total parameter counts and the LoRA compression ratio."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = trainable / total * 100
    print(f"Trainable parameters : {trainable:>15,}  ({ratio:.4f}% of total)")
    print(f"Total parameters     : {total:>15,}")
    print(f"Frozen parameters    : {total - trainable:>15,}")


def get_gpu_memory_stats() -> None:
    """Print current VRAM allocation if CUDA is available, otherwise report CPU mode."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
        print(f"VRAM total : {total_vram:.1f} GB")
        print(f"VRAM alloc : {allocated:.2f} GB  |  reserved: {reserved:.2f} GB")
    else:
        print("Running in CPU mode — VRAM stats not available")
        print("Tip: set MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' for faster CPU runs")


def load_alpaca_subset(num_samples: int = MAX_SAMPLES) -> Dataset:
    """Download and return the first `num_samples` records of tatsu-lab/alpaca."""
    print(f"Loading {num_samples} samples from {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME, split="train")
    return ds.select(range(min(num_samples, len(ds))))


# ── Student-implemented stubs ──────────────────────────────────────────────────

def configure_qlora_quantization() -> BitsAndBytesConfig:
    """
    Return a BitsAndBytesConfig that loads the base model in 4-bit NF4 precision.

    Required settings:
      - load_in_4bit=True              → quantize base weights to 4-bit
      - bnb_4bit_compute_dtype=torch.bfloat16  → upcast during computation
      - bnb_4bit_use_double_quant=True → quantize the quantization constants too
      - bnb_4bit_quant_type="nf4"      → Normal Float 4 (best for weight distributions)

    See lesson §QLoRA — the theory section explains why NF4 levels cluster near zero
    to match the roughly Gaussian distribution of pretrained weights.
    """
    raise NotImplementedError("TODO: implement this")


def load_base_model_and_tokenizer(
    model_id: str,
    bnb_config: BitsAndBytesConfig,
) -> tuple[Any, Any]:
    """
    Load the causal LM and its tokenizer, quantized to 4-bit.

    Steps:
      1. Load tokenizer: AutoTokenizer.from_pretrained(model_id).
         Set tokenizer.pad_token = tokenizer.eos_token
         (Llama has no dedicated pad token by default).
      2. Load model: AutoModelForCausalLM.from_pretrained(
             model_id,
             quantization_config=bnb_config,
             device_map="auto",
         )
      3. Call prepare_model_for_kbit_training(model).
         This enables gradient checkpointing and casts layer norms to float32,
         which prevents numeric instability during the backward pass.
      4. Return (model, tokenizer).

    After this function runs, get_gpu_memory_stats() should show ≥3 GB allocated
    (GPU) or print "CPU mode" (expected for TinyLlama CPU fallback).
    """
    raise NotImplementedError("TODO: implement this")


def build_lora_config() -> LoraConfig:
    """
    Return a LoraConfig that injects low-rank adapters into the attention projections.

    Required parameters:
      r             = LORA_RANK        (16)    — adapter rank
      lora_alpha    = LORA_ALPHA       (32)    — scaling factor α
      lora_dropout  = LORA_DROPOUT     (0.05)  — dropout on adapter path
      target_modules = ["q_proj", "v_proj"]   — inject into query and value only
      bias          = "none"                   — do not train bias terms
      task_type     = TaskType.CAUSAL_LM       — autoregressive generation task

    Why q_proj and v_proj only? Adding k_proj and o_proj gives marginal quality
    improvement at 2× the adapter parameter cost — rank 16 on q+v is the standard
    starting point for instruction-tuning (Hu et al. 2022, Table 5).
    """
    raise NotImplementedError("TODO: implement this")


def format_training_record(record: dict[str, str]) -> str:
    """
    Convert one Alpaca dataset record into an instruction-following training string.

    Alpaca format with non-empty input:
        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {output}

    Alpaca format with empty input (omit the Input block entirely):
        ### Instruction:
        {instruction}

        ### Response:
        {output}

    The model trains on the full string including the response — this is supervised
    fine-tuning (SFT), not RLHF. Return the complete formatted string.

    Hint: check `record["input"].strip()` to decide whether to include the block.
    """
    raise NotImplementedError("TODO: implement this")


def run_fine_tuning(
    model: Any,
    tokenizer: Any,
    lora_config: LoraConfig,
    dataset: Dataset,
) -> SFTTrainer:
    """
    Wrap the model with LoRA adapters and run supervised fine-tuning for TRAINING_STEPS.

    Steps:
      1. peft_model = get_peft_model(model, lora_config)
         This freezes all base weights and adds trainable A, B matrices.
      2. Print parameter counts: count_trainable_parameters(peft_model).
      3. Build TrainingArguments:
           output_dir                = OUTPUT_DIR
           max_steps                 = TRAINING_STEPS
           per_device_train_batch_size = 1
           gradient_accumulation_steps = 4   ← effective batch size = 4
           learning_rate             = 2e-4
           fp16                      = torch.cuda.is_available()
           logging_steps             = 10
           save_steps                = 50
           optim                     = "paged_adamw_8bit"  ← QLoRA paged optimizer
           lr_scheduler_type         = "cosine"
      4. Map format_training_record over the dataset to add a "text" column:
           formatted = dataset.map(lambda ex: {"text": format_training_record(ex)})
      5. trainer = SFTTrainer(
             model=peft_model,
             args=training_args,
             train_dataset=formatted,
             dataset_text_field="text",
             max_seq_length=MAX_SEQ_LEN,
             tokenizer=tokenizer,
         )
      6. trainer.train()
      7. Return trainer.

    The paged_adamw_8bit optimizer pages optimizer states to CPU RAM during
    backward-pass memory spikes, preventing OOM on consumer GPUs (see lesson §QLoRA).
    """
    raise NotImplementedError("TODO: implement this")


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 150,
) -> str:
    """
    Run greedy decoding on `prompt` and return only the newly generated tokens.

    Steps:
      1. inputs = tokenizer(prompt, return_tensors="pt")
         Move each tensor in inputs to model.device:
           inputs = {k: v.to(model.device) for k, v in inputs.items()}
      2. with torch.no_grad():
             output_ids = model.generate(
                 **inputs,
                 max_new_tokens=max_new_tokens,
                 do_sample=False,                     ← greedy, deterministic
                 pad_token_id=tokenizer.eos_token_id,
             )
      3. Decode only the new tokens (strip the prompt prefix):
           prompt_len = inputs["input_ids"].shape[1]
           new_ids    = output_ids[0][prompt_len:]
           return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    Returning only the continuation (not the echoed prompt) makes before/after
    comparisons readable without post-processing.
    """
    raise NotImplementedError("TODO: implement this")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Node 4.2 — QLoRA Fine-Tuning Pipeline")
    print("=" * 60)
    print(f"Model  : {MODEL_ID}")
    print(f"Device : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    # ── Hardware snapshot ──────────────────────────────────────────────────────
    get_gpu_memory_stats()
    print()

    # ── Task 1: quantization config ───────────────────────────────────────────
    bnb_config = configure_qlora_quantization()
    print("QLoRA quantization config created.")
    print(f"  4-bit NF4   : {bnb_config.load_in_4bit}")
    print(f"  Double quant: {bnb_config.bnb_4bit_use_double_quant}")
    print()

    # ── Task 2: load base model ───────────────────────────────────────────────
    print(f"Loading {MODEL_ID} (this may take several minutes)...")
    model, tokenizer = load_base_model_and_tokenizer(MODEL_ID, bnb_config)
    print("Model loaded.")
    get_gpu_memory_stats()
    print()

    # ── Task 3: LoRA config ───────────────────────────────────────────────────
    lora_config = build_lora_config()
    print(
        f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
        f"targets={list(lora_config.target_modules)}"
    )
    print()

    # ── Task 4: verify format (dataset loaded by helper, format stub tested) ──
    dataset = load_alpaca_subset(MAX_SAMPLES)
    sample_formatted = format_training_record(dataset[0])
    print("Sample formatted record (first 300 chars):")
    print(sample_formatted[:300])
    if len(sample_formatted) > 300:
        print("...")
    print()

    # ── Task 6 (before): base model generations ───────────────────────────────
    test_prompts = [
        (
            "### Instruction:\n"
            "Explain what a transformer neural network is.\n\n"
            "### Response:\n"
        ),
        (
            "### Instruction:\n"
            "Write a Python function that reverses a string.\n\n"
            "### Response:\n"
        ),
        (
            "### Instruction:\n"
            "What is the difference between supervised and unsupervised learning?\n\n"
            "### Response:\n"
        ),
    ]

    print("─" * 60)
    print("--- Base model generations (before fine-tuning) ---")
    print("─" * 60)
    base_outputs: list[str] = []
    for prompt in test_prompts:
        response = generate_response(model, tokenizer, prompt)
        base_outputs.append(response)
        instruction_line = prompt.splitlines()[1]
        print(f"Prompt   : {instruction_line}")
        print(f"Base out : {response[:200]}")
        print()

    # ── Task 5: fine-tuning ───────────────────────────────────────────────────
    print(f"Starting fine-tuning for {TRAINING_STEPS} steps...")
    trainer = run_fine_tuning(model, tokenizer, lora_config, dataset)
    print("Fine-tuning complete.")
    get_gpu_memory_stats()
    print()

    # ── Task 6 (after): fine-tuned model generations ─────────────────────────
    print("─" * 60)
    print("--- Fine-tuned model generations (after LoRA) ---")
    print("─" * 60)
    for i, prompt in enumerate(test_prompts):
        response = generate_response(trainer.model, tokenizer, prompt)
        instruction_line = prompt.splitlines()[1]
        print(f"Prompt   : {instruction_line}")
        print(f"Base out : {base_outputs[i][:200]}")
        print(f"LoRA out : {response[:200]}")
        print("─" * 60)


if __name__ == "__main__":
    main()
