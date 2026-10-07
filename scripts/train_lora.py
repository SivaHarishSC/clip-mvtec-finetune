"""
LoRA fine-tune CLIP ViT-B/32 on MVTec-AD image-text pairs (templated
captions) using the standard CLIP contrastive objective. Base weights are
frozen; only LoRA adapters on the attention projections train.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from clip_mvtec.dataset import CaptionedMVTecDataset, make_collate_fn
from clip_mvtec.model import apply_lora, clip_contrastive_loss, load_base_model_and_processor

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def evaluate_val_loss(model, loader, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            loss = clip_contrastive_loss(out.logits_per_image)
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions-csv", default=str(PROJECT_ROOT / "data" / "captions.csv"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "checkpoints" / "clip-lora-mvtec"))
    parser.add_argument("--mlflow-experiment", default="clip-lora-mvtec")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor = load_base_model_and_processor(device)
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model.to(device)
    model.print_trainable_parameters()

    collate_fn = make_collate_fn(processor)
    train_ds = CaptionedMVTecDataset(args.captions_csv, str(PROJECT_ROOT), split="train")
    val_ds = CaptionedMVTecDataset(args.captions_csv, str(PROJECT_ROOT), split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    mlflow.set_experiment(args.mlflow_experiment)
    best_val_loss = float("inf")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
            "train_size": len(train_ds), "val_size": len(val_ds),
        })

        global_step = 0
        for epoch in range(args.epochs):
            model.train()
            epoch_start = time.time()
            running_loss = 0.0
            for batch in train_loader:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                    loss = clip_contrastive_loss(out.logits_per_image)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                global_step += 1
                if global_step % 20 == 0:
                    mlflow.log_metric("train_loss_step", loss.item(), step=global_step)

            train_loss = running_loss / len(train_loader)
            val_loss = evaluate_val_loss(model, val_loader, device)
            epoch_time = time.time() - epoch_start

            mlflow.log_metrics({
                "train_loss_epoch": train_loss,
                "val_loss_epoch": val_loss,
                "epoch_time_sec": epoch_time,
            }, step=epoch)
            print(f"epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({epoch_time:.1f}s)")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(args.output_dir)
                print(f"  saved new best checkpoint (val_loss={val_loss:.4f}) -> {args.output_dir}")

        mlflow.log_metric("best_val_loss", best_val_loss)
        if device == "cuda":
            mlflow.log_metric("peak_vram_mb", torch.cuda.max_memory_allocated() / 1e6)

    print("Training complete. Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()
