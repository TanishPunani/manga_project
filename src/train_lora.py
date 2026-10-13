"""Sequence-level LoRA SFT trainer for Qwen2.5-VL manga transcription.

This script trains on the sequence IDs in experiment_split/train_ids.json.
It masks the user prompt from the language-model loss and saves the adapter
and a JSON training report. Run on a GPU with sufficient VRAM.
"""
import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="outputs/reproducible_lora")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.root).resolve()
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)

    manifest = read_jsonl(root / "sft_manifest.jsonl")
    train_ids = json.loads(
        (root / "experiment_split/train_ids.json").read_text(encoding="utf-8")
    )
    by_id = {row["sequence_id"]: row for row in manifest}
    records = [by_id[sid] for sid in train_ids if sid in by_id]
    if not records:
        raise ValueError("No training records matched train_ids.json")

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        min_pixels=64 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )

    # Avoid FP16 NaNs observed in some manga pages' vision activations.
    model.visual.float()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    report = {
        "model_name": args.model_name,
        "seed": args.seed,
        "learning_rate": args.lr,
        "requested_steps": args.steps,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "records_available": len(records),
        "losses": [],
        "note": "New reproducibility run; not the original interactive run.",
    }

    for step in range(args.steps):
        row = records[step % len(records)]
        images = []
        for rel in row["images"]:
            path = root / rel
            if not path.exists():
                raise FileNotFoundError(path)
            images.append(Image.open(path).convert("RGB"))

        user_content = [{"type": "text", "text": row["prompt"]}]
        user_content.extend(
            {"type": "image", "image": image} for image in images
        )
        user_message = {"role": "user", "content": user_content}
        answer = json.dumps(row["target"], ensure_ascii=False)
        full_messages = [
            user_message,
            {"role": "assistant", "content": answer},
        ]

        prompt_text = processor.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        full_text = processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        image_inputs, video_inputs = process_vision_info(full_messages)
        batch = processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Keep visual input precision consistent with model.visual.float().
        if "pixel_values" in batch:
            batch["pixel_values"] = batch["pixel_values"].float()

        # Tokenize the prompt separately so its tokens do not contribute to loss.
        prompt_images, prompt_videos = process_vision_info([user_message])
        prompt_batch = processor(
            text=[prompt_text],
            images=prompt_images,
            videos=prompt_videos,
            padding=True,
            return_tensors="pt",
        )
        prompt_length = prompt_batch["input_ids"].shape[1]

        labels = batch["input_ids"].clone()
        labels[:, :prompt_length] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels

        optimizer.zero_grad(set_to_none=True)
        result = model(**batch)
        loss = result.loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step + 1}, sequence {row['sequence_id']}"
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        value = float(loss.detach().cpu())
        report["losses"].append({
            "step": step + 1,
            "sequence_id": row["sequence_id"],
            "loss": value,
        })
        print(f"Step {step + 1}/{args.steps} | {row['sequence_id']} | loss={value:.4f}")

        if (step + 1) % 8 == 0:
            checkpoint = output / f"checkpoint_step_{step + 1}"
            model.save_pretrained(checkpoint)
            processor.save_pretrained(checkpoint)
            (output / "training_report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )

    model.save_pretrained(output)
    processor.save_pretrained(output)
    report["completed_steps"] = len(report["losses"])
    report["mean_loss"] = (
        sum(item["loss"] for item in report["losses"]) / len(report["losses"])
    )
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Saved adapter and report to: {output}")
    print(f"Completed steps: {report['completed_steps']}")
    print(f"Mean training loss: {report['mean_loss']:.4f}")


if __name__ == "__main__":
    main()
