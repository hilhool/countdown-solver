"""Fine-tune google/gemma-3-1b-it to produce Countdown solution traces.

    python 2_train.py            # reads train_traces.jsonl -> ./gemma-countdown

LoRA SFT with completion-only loss: the prompt is masked, so the model is only
trained on the trace it should produce. Prompt/response format:

    <start_of_turn>user
    Numbers: 75 80 90 24
    Target: 61<end_of_turn>
    <start_of_turn>model
    90 - 80 = 10
    75 - 24 = 51
    10 + 51 = 61<end_of_turn>
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
import random

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from countdown import make_prompt_numbers

MODEL = "google/gemma-3-1b-it"
INPUT_FILE = "train_traces.jsonl"
OUTPUT_DIR = "./gemma-countdown"
NUM_SAMPLES = 200_000

EPOCHS = 3                     # best checkpoint is restored at the end, so extra
                               # epochs only help if eval_loss keeps improving
BATCH_SIZE = 4
GRAD_ACCUM = 8                 # effective batch = 32
LR = 2e-4
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.05
MAX_SEQ_LEN = 160              # traces are multi-line, longer than bare equations

LORA_R = 128
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

EVAL_SPLIT = 0.002
EVAL_STEPS = 1000
SAVE_STEPS = 1000
LOGGING_STEPS = 50

RESPONSE_MARKER = "<start_of_turn>model\n"


class LossLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            eval_loss = logs.get("eval_loss")
            tail = f", eval_loss={eval_loss:.4f}" if eval_loss is not None else ""
            print(f">>> step {state.global_step}: loss={logs['loss']:.4f}"
                  f", grad_norm={logs.get('grad_norm', float('nan')):.3f}{tail}", flush=True)


def load_records(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
    return records


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = load_records(INPUT_FILE)[:NUM_SAMPLES]
    random.seed(42)
    random.shuffle(records)
    print(f"Loaded {len(records):,} records")

    split = int(len(records) * (1 - EVAL_SPLIT))
    train_records, eval_records = records[:split], records[split:]

    def to_text(rec):
        messages = [
            {"role": "user", "content": make_prompt_numbers(rec["numbers"], rec["target"])},
            {"role": "assistant", "content": rec["trace"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def response_start(input_ids):
        text = tokenizer.decode(input_ids, skip_special_tokens=False)
        pos = text.rfind(RESPONSE_MARKER)
        if pos == -1:
            return 0
        prefix = text[: pos + len(RESPONSE_MARKER)]
        return len(tokenizer.encode(prefix, add_special_tokens=False))

    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LEN, padding=False)
        labels = []
        for ids in out["input_ids"]:
            masked = ids.copy()
            cut = response_start(ids)
            masked[:cut] = [-100] * cut
            labels.append(masked)
        out["labels"] = labels
        return out

    train_ds = Dataset.from_dict({"text": [to_text(r) for r in train_records]}) \
        .map(tokenize, batched=True, remove_columns=["text"])
    eval_ds = Dataset.from_dict({"text": [to_text(r) for r in eval_records]}) \
        .map(tokenize, batched=True, remove_columns=["text"])

    sample = train_ds[0]["labels"]
    trainable = sum(1 for l in sample if l != -100)
    print(f"Sample: {len(sample)} tokens, {trainable} trained (rest masked prompt)")

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype, device_map="cuda")
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    ))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        gradient_checkpointing=True,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[LossLogger()],
    )

    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")] \
        if os.path.exists(OUTPUT_DIR) else []
    trainer.train(resume_from_checkpoint=bool(checkpoints))

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
