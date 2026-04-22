import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

"""
Script 2: Fine-tune Gemma-3-1B-IT on synthetic data (SFT + LoRA)
Run:    python 2_train.py
Output: ./gemma-countdown/

New minimal prompt format for better 1B model learning:
<start_of_turn>user
Numbers: 75 80 90 24
Target: 61<end_of_turn>
<start_of_turn>model
90 - 80 + 75 - 24<end_of_turn>
"""

import json
import random
import time
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

# ─── CONFIG ───────────────────────────────────────────────────────────────────
STUDENT_MODEL = "google/gemma-3-1b-it"
INPUT_FILE = "train_verified.jsonl"
OUTPUT_DIR = "./gemma-countdown"
NUM_SAMPLES = 150000        # Balance quality vs speed

# Training hyperparameters
EPOCHS = 2
BATCH_SIZE = 4
GRAD_ACCUM = 8              # effective batch = 32
LR = 2e-4
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.05
MAX_SEQ_LEN = 96            # Shorter prompts need less context
GRADIENT_CHECKPOINTING = False  # Disable to test VRAM

# LoRA config
LORA_R = 128
LORA_ALPHA = 128

# Loss masking
COMPLETION_ONLY = True      # Only compute loss on equation (response)

# Eval split
EVAL_SPLIT = 0.002          # ~200 examples for validation
EVAL_STEPS = 1000           # Evaluate every N steps

# Logging
LOGGING_STEPS = 50
SAVE_STRATEGY = "steps"
SAVE_STEPS = 1000
# ──────────────────────────────────────────────────────────────────────────────


class LossCallback(TrainerCallback):
    """Print loss and grad_norm at each logging step."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            loss = logs.get("loss", "N/A")
            eval_loss = logs.get("eval_loss", "")
            grad_norm = logs.get("grad_norm", "N/A")
            epoch = logs.get("epoch", "N/A")
            eval_str = f", eval_loss={eval_loss}" if eval_loss else ""
            print(f"\n>>> Step {step}: loss={loss}, grad_norm={grad_norm}, epoch={epoch}{eval_str}", flush=True)


def load_data(path: str) -> list[dict]:
    """Load JSONL data in new format: {numbers, target, equation}."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # Support both old format (with 'nums' and 'prompt') and new format
            if "numbers" in rec:
                records.append(rec)
            elif "nums" in rec:
                records.append({
                    "numbers": rec["nums"],
                    "target": rec["target"],
                    "equation": rec["equation"],
                })
    print(f"  Loaded {len(records)} examples")
    return records


def make_prompt(numbers: list, target: int) -> str:
    """Create minimal prompt for training/inference."""
    nums_str = " ".join(map(str, numbers))
    return f"Numbers: {nums_str}\nTarget: {target}"


def format_sample(record: dict, tokenizer) -> str:
    """Create full training sample with chat template."""
    prompt = make_prompt(record["numbers"], record["target"])
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": record["equation"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )


def find_response_start(input_ids: list, tokenizer) -> int:
    """Find position where assistant response starts.

    Gemma-3 uses "<start_of_turn>model\n" to mark assistant turn.
    """
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

    # Find the LAST occurrence of model turn marker
    marker = "<start_of_turn>model\n"
    marker_pos = full_text.rfind(marker)
    if marker_pos == -1:
        marker = "<start_of_turn>model"
        marker_pos = full_text.rfind(marker)

    if marker_pos == -1:
        return 0  # Fallback: train on everything

    # Count tokens before response content
    text_before_response = full_text[:marker_pos + len(marker)]
    tokens_before = tokenizer.encode(text_before_response, add_special_tokens=False)
    return len(tokens_before)


def main():
    print("=" * 60)
    print("Step 2: Fine-tune Gemma-3-1B-IT (SFT + LoRA)")
    print("=" * 60)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer ({STUDENT_MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ───────────────────────────────────────────────────────────
    print(f"\nLoading data from {INPUT_FILE}...")
    records = load_data(INPUT_FILE)
    records = records[:NUM_SAMPLES]  # Take first 100k
    print(f"  Dataset size: {len(records)} examples")

    # Shuffle before split to avoid distribution bias
    random.seed(42)
    random.shuffle(records)

    # Create train/eval split
    split_idx = int(len(records) * (1 - EVAL_SPLIT))
    train_records = records[:split_idx]
    eval_records = records[split_idx:]
    print(f"  Train: {len(train_records)}, Eval: {len(eval_records)}")

    # Format samples
    train_texts = [format_sample(r, tokenizer) for r in train_records]
    eval_texts = [format_sample(r, tokenizer) for r in eval_records]

    # Tokenize with completion-only loss masking
    def tokenize_and_mask(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
        )

        all_labels = []
        for input_ids in tokenized["input_ids"]:
            labels = input_ids.copy()
            if COMPLETION_ONLY:
                response_start = find_response_start(input_ids, tokenizer)
                labels[:response_start] = [-100] * response_start
            all_labels.append(labels)

        tokenized["labels"] = all_labels
        return tokenized

    train_dataset = Dataset.from_dict({"text": train_texts})
    train_dataset = train_dataset.map(tokenize_and_mask, batched=True, remove_columns=["text"])

    eval_dataset = Dataset.from_dict({"text": eval_texts})
    eval_dataset = eval_dataset.map(tokenize_and_mask, batched=True, remove_columns=["text"])

    # ── Sample verification ───────────────────────────────────────────────
    print(f"\n  --- Sample formatted text ---")
    print(f"  {train_texts[0]}")
    print(f"  --- end sample ---\n")

    # Verify masking
    sample_labels = train_dataset[0]["labels"]
    masked_count = sum(1 for l in sample_labels if l == -100)
    total_count = len(sample_labels)
    loss_mode = "completion-only" if COMPLETION_ONLY else "full-sequence"
    print(f"  Loss mode: {loss_mode}")
    print(f"    Total tokens: {total_count}")
    print(f"    Masked (ignored): {masked_count}")
    print(f"    Trainable: {total_count - masked_count}")
    print(f"    Loss computed on: {100*(total_count - masked_count)/total_count:.1f}% of tokens")

    # ── Estimate training time ────────────────────────────────────────────
    total_steps = (len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS
    print(f"\n  Estimated training:")
    print(f"    Total steps: ~{total_steps:,}")
    print(f"    At ~2 steps/sec: ~{total_steps/2/3600:.1f} hours")

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\nLoading {STUDENT_MODEL}...")

    # Check bf16 support
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
        use_bf16 = True
        print("  Using bf16 (Ampere+ GPU detected)")
    else:
        dtype = torch.float16
        use_bf16 = False
        print("  Using fp16")

    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=dtype,
        device_map="cuda",
    )

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()  # Save VRAM
    model.print_trainable_parameters()

    # ── Training config ───────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
        report_to="none",
        dataloader_num_workers=0,  # Windows compatibility
        remove_unused_columns=False,
    )

    print(f"\nTraining config:")
    print(f"  Epochs:          {EPOCHS}")
    print(f"  LR:              {LR} ({LR_SCHEDULER})")
    print(f"  Warmup:          {WARMUP_RATIO}")
    print(f"  LoRA r/alpha:    {LORA_R}/{LORA_ALPHA}")
    print(f"  Effective batch: {BATCH_SIZE}×{GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM}")
    print(f"  Max seq length:  {MAX_SEQ_LEN}")
    print(f"  Loss mode:       {loss_mode}")
    print(f"  Gradient ckpt:   True")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[LossCallback()],
    )

    # Check for existing checkpoints to resume training
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")] if os.path.exists(OUTPUT_DIR) else []
    resume = len(checkpoints) > 0
    print(f"Resume from checkpoint: {resume}")
    trainer.train(resume_from_checkpoint=resume)

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\nSaving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nTraining complete!")
    print("→ Next: python 3_inference.py")


if __name__ == "__main__":
    main()
