# Assignment 4.2 — Fine-Tune Llama-3 on Domain Dataset

## Objective

Build a complete QLoRA fine-tuning pipeline that loads Meta Llama-3-8B in 4-bit NF4 precision, attaches LoRA adapters to its attention projections, trains for 50 steps on an instruction dataset, and demonstrates measurable parameter efficiency relative to full fine-tuning.

## Background

In the lesson you derived the LoRA forward pass $h = xW + (\alpha/r) \cdot xBA$, showed that rank-16 adapters on a 4096×4096 projection use 0.39% of the full fine-tune parameter budget, and walked through how QLoRA stacks 4-bit NF4 quantization and paged optimizers on top to fit a 7B model in ~5.6 GB VRAM. This assignment implements that pipeline end-to-end using `peft`, `transformers`, `bitsandbytes`, and `trl`. You will compare text generations from the base checkpoint against the LoRA-adapted model to see the behavioral shift after just 50 steps.

> **Model access:** Meta Llama-3-8B is gated. Accept the terms at
> `https://huggingface.co/meta-llama/Meta-Llama-3-8B` then authenticate with
> `huggingface-cli login`.
>
> **CPU / low-VRAM fallback:** Set `MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"` at the
> top of `starter.py` — no token required, runs on CPU in ~20 minutes.

## Setup

```bash
pip install "torch>=2.1" "transformers>=4.40" "peft>=0.10" \
            "datasets>=2.18" "bitsandbytes>=0.43" "trl>=0.8"
```

Log in to Hugging Face (skip for TinyLlama fallback):

```bash
huggingface-cli login
```

## Tasks

Work through the six stubs in `starter/starter.py` in order. Each stub corresponds to one numbered task below.

### Task 1 — Configure 4-bit QLoRA quantization

Implement `configure_qlora_quantization()` to return a `BitsAndBytesConfig` that:

- Enables 4-bit loading (`load_in_4bit=True`)
- Sets compute dtype to `torch.bfloat16`
- Enables double quantization (`bnb_4bit_use_double_quant=True`)
- Sets the quantization type to `"nf4"`

Run `main()` — you should see the config attributes printed before any model download begins.

### Task 2 — Load the quantized base model

Implement `load_base_model_and_tokenizer(model_id, bnb_config)` to:

1. Load the tokenizer with `AutoTokenizer.from_pretrained(model_id)`. Set `tokenizer.pad_token = tokenizer.eos_token` (Llama has no default pad token).
2. Load the model with `AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")`.
3. Call `prepare_model_for_kbit_training(model)` — this enables gradient checkpointing and casts layer norms to fp32 for numeric stability.
4. Return `(model, tokenizer)`.

After this task, `get_gpu_memory_stats()` should show VRAM allocated ≥ 3 GB (GPU) or print CPU mode.

### Task 3 — Build the LoRA configuration

Implement `build_lora_config()` to return a `LoraConfig` with:

| Parameter | Value |
|---|---|
| `r` | `LORA_RANK` (16) |
| `lora_alpha` | `LORA_ALPHA` (32) |
| `lora_dropout` | `LORA_DROPOUT` (0.05) |
| `target_modules` | `["q_proj", "v_proj"]` |
| `bias` | `"none"` |
| `task_type` | `TaskType.CAUSAL_LM` |

Targeting `q_proj` and `v_proj` gives the best quality-per-parameter trade-off for instruction-following tasks (see lesson §LoRA).

### Task 4 — Format Alpaca instruction records

Implement `format_training_record(record)` to convert one dataset dict into an Alpaca-style instruction string:

```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

Omit the `### Input:` block entirely when `record["input"]` is an empty string — the model should not learn to expect an empty input section. Return the complete string including the response.

### Task 5 — Run the fine-tuning loop

Implement `run_fine_tuning(model, tokenizer, lora_config, dataset)` to:

1. Wrap the model: `peft_model = get_peft_model(model, lora_config)`.
2. Print trainable vs total parameters using `count_trainable_parameters(peft_model)`.
3. Create `TrainingArguments` with the values below (all others at default):

   | Argument | Value |
   |---|---|
   | `output_dir` | `OUTPUT_DIR` |
   | `max_steps` | `TRAINING_STEPS` (50) |
   | `per_device_train_batch_size` | `1` |
   | `gradient_accumulation_steps` | `4` |
   | `learning_rate` | `2e-4` |
   | `fp16` | `True` if CUDA available, else `False` |
   | `logging_steps` | `10` |
   | `save_steps` | `50` |
   | `optim` | `"paged_adamw_8bit"` |
   | `lr_scheduler_type` | `"cosine"` |

4. Map `format_training_record` over `dataset` to produce a `"text"` column.
5. Construct `SFTTrainer(model=peft_model, args=training_args, train_dataset=formatted_dataset, dataset_text_field="text", max_seq_length=MAX_SEQ_LEN, tokenizer=tokenizer)`.
6. Call `trainer.train()` and return the trainer.

### Task 6 — Compare base vs fine-tuned generations

Implement `generate_response(model, tokenizer, prompt, max_new_tokens=150)` to:

1. Tokenize `prompt` with `tokenizer(prompt, return_tensors="pt")` and move inputs to `model.device`.
2. Call `model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)`.
3. Decode `outputs[0]` with `tokenizer.decode(..., skip_special_tokens=True)`.
4. Strip the prompt prefix from the decoded string and return only the newly generated portion.

`main()` calls `generate_response()` on three test prompts before fine-tuning (using the quantized base model) and again after (using `trainer.model`), then prints both side by side.

## Expected Output

GPU run (Llama-3-8B, RTX 4090):

```
============================================================
Node 4.2 — QLoRA Fine-Tuning Pipeline
============================================================
Model  : meta-llama/Meta-Llama-3-8B
Device : cuda

GPU        : NVIDIA GeForce RTX 4090
VRAM total : 24.0 GB
VRAM alloc : 0.00 GB  |  reserved: 0.00 GB

QLoRA quantization config created.
  4-bit NF4   : True
  Double quant: True

Loading meta-llama/Meta-Llama-3-8B (this may take several minutes)...
Model loaded.
VRAM alloc : 3.82 GB  |  reserved: 3.95 GB

LoRA config: r=16, alpha=32, targets=['q_proj', 'v_proj']

Loading 1000 samples from tatsu-lab/alpaca...

--- Base model generations (before fine-tuning) ---
Prompt   : ### Instruction:\nExplain what a transformer neural network is...
Base out : [unformatted continuation, may repeat or drift]
...

Starting fine-tuning for 50 steps...
Trainable parameters :       13,631,488  (0.1638% of total)
Total parameters     :    8,323,407,872
Frozen parameters    :    8,309,776,384
{'loss': 2.1xxx, 'learning_rate': 1.9e-04, 'epoch': 0.0x}
{'loss': 1.9xxx, 'learning_rate': 1.7e-04, 'epoch': 0.1x}
...
{'loss': 1.7xxx, 'learning_rate': ...}
Fine-tuning complete.

--- Fine-tuned model generations (after 50 LoRA steps) ---
Prompt   : ### Instruction:\nExplain what a transformer neural network is...
LoRA out : [more structured, on-format response]
...
```

CPU run (TinyLlama-1.1B):

```
Running in CPU mode — VRAM stats not available
Trainable parameters :        4,456,448  (0.4035% of total)
Total parameters     :    1,100,048,384
Training loss should fall from ~2.4 to ~1.9 over 50 steps (~20 min on CPU)
```

Acceptable loss range after 50 steps: **1.6 – 2.4** (varies with model, hardware, seed). Any monotonically decreasing trend over the logged steps is correct.

## Evaluation Criteria

- [ ] `configure_qlora_quantization()` returns a `BitsAndBytesConfig` with `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, and `bnb_4bit_use_double_quant=True`
- [ ] `load_base_model_and_tokenizer()` sets `pad_token` and calls `prepare_model_for_kbit_training()`
- [ ] `build_lora_config()` targets `["q_proj", "v_proj"]` with `r=16` and `task_type=TaskType.CAUSAL_LM`
- [ ] `format_training_record()` omits the `### Input:` block when the input field is empty
- [ ] `count_trainable_parameters()` output shows trainable parameters < 1% of total for Llama-3-8B (or < 0.5% for TinyLlama)
- [ ] Training runs for exactly `TRAINING_STEPS` steps without error and logs loss every 10 steps
- [ ] `generate_response()` returns only the generated continuation — not the full decoded sequence including the prompt
- [ ] The before-fine-tuning and after-fine-tuning generations are visibly different in format or content for at least one of the three prompts

## Extension Challenge

**Rank sensitivity experiment:** Run the full pipeline three times with `LORA_RANK` set to `4`, `16`, and `64`. For each run record:

- Final training loss at step 50
- Trainable parameter count
- One generation on the same fixed prompt

Plot loss vs rank and generation quality vs rank. Write a 3-sentence analysis of the trade-off: at what rank does quality plateau relative to parameter cost? No starter code provided — you must modify `main()` to loop over ranks and collect results into a comparison table.
