"""
Benchmark a merged CLIP checkpoint in either full precision or bitsandbytes
8-bit quantized form: inference latency (ms/query), peak VRAM, GPU
temperature before/after (to catch thermal throttling), and Recall@K on the
test split.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from clip_mvtec.retrieval_eval import embed_all, recall_at_k

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def gpu_temp_c():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().splitlines()[0])
    except Exception as e:
        return None


def load_rows(captions_csv, split):
    with open(captions_csv, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def load_model(model_dir, quantization, device):
    from transformers import CLIPModel, CLIPProcessor
    processor = CLIPProcessor.from_pretrained(model_dir)
    if quantization == "8bit":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = CLIPModel.from_pretrained(model_dir, quantization_config=bnb_config, device_map={"": 0})
    else:
        model = CLIPModel.from_pretrained(model_dir).to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def measure_latency(model, processor, rows, device, n_batches=20, batch_size=8, warmup=5):
    from PIL import Image
    images = [Image.open(PROJECT_ROOT / r["image_path"]).convert("RGB") for r in rows[:batch_size]]
    captions = [r["caption"] for r in rows[:batch_size]]
    enc_img = processor(images=images, return_tensors="pt").to(device)
    enc_txt = processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(device)

    for _ in range(warmup):
        model.get_image_features(pixel_values=enc_img["pixel_values"])
        model.get_text_features(input_ids=enc_txt["input_ids"], attention_mask=enc_txt["attention_mask"])
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_batches):
        model.get_image_features(pixel_values=enc_img["pixel_values"])
        model.get_text_features(input_ids=enc_txt["input_ids"], attention_mask=enc_txt["attention_mask"])
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_queries = n_batches * batch_size * 2  # image + text encode per item
    ms_per_query = (elapsed / total_queries) * 1000
    return ms_per_query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(PROJECT_ROOT / "checkpoints" / "clip-mvtec-merged"))
    parser.add_argument("--quantization", choices=["none", "8bit"], default="none")
    parser.add_argument("--captions-csv", default=str(PROJECT_ROOT / "data" / "captions.csv"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and args.quantization == "8bit":
        raise SystemExit("bitsandbytes 8-bit quantization requires CUDA.")

    temp_before = gpu_temp_c()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    model, processor = load_model(args.model_dir, args.quantization, device)
    rows = load_rows(args.captions_csv, args.split)

    ms_per_query = measure_latency(model, processor, rows, device)
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None

    image_embeds, text_embeds, image_caption_idx, unique_captions = embed_all(
        model, processor, rows, PROJECT_ROOT, device
    )
    recalls = recall_at_k(image_embeds, text_embeds, image_caption_idx)

    temp_after = gpu_temp_c()

    result = {
        "label": args.label,
        "quantization": args.quantization,
        "split": args.split,
        "n_images": len(rows),
        "latency_ms_per_query": ms_per_query,
        "peak_vram_mb": peak_vram_mb,
        "gpu_temp_c_before": temp_before,
        "gpu_temp_c_after": temp_after,
        "thermal_throttle_suspected": (temp_after - temp_before > 15) if (temp_before and temp_after) else None,
        "recall": recalls,
    }
    print(json.dumps(result, indent=2))

    out_path = Path(args.out) if args.out else PROJECT_ROOT / "results" / f"benchmark_{args.label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
