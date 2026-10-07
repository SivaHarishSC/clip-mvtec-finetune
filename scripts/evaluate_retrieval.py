"""
Evaluate Recall@{1,5,10} for image<->text retrieval on a captions.csv split,
for either zero-shot base CLIP or a LoRA fine-tuned checkpoint.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from clip_mvtec.model import BASE_MODEL_ID, BASE_MODEL_REVISION
from clip_mvtec.retrieval_eval import embed_all, recall_at_k

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_rows(captions_csv, split):
    with open(captions_csv, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions-csv", default=str(PROJECT_ROOT / "data" / "captions.csv"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--lora-adapter", default=None, help="Path to a saved PEFT adapter; omit for zero-shot base CLIP")
    parser.add_argument("--label", default="base", help="Label for this run in the output JSON")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(BASE_MODEL_ID, revision=BASE_MODEL_REVISION).to(device)
    processor = CLIPProcessor.from_pretrained(BASE_MODEL_ID, revision=BASE_MODEL_REVISION)

    if args.lora_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_adapter)
        model = model.merge_and_unload()
    model.eval()

    rows = load_rows(args.captions_csv, args.split)
    image_embeds, text_embeds, image_caption_idx, unique_captions = embed_all(
        model, processor, rows, PROJECT_ROOT, device
    )
    recalls = recall_at_k(image_embeds, text_embeds, image_caption_idx)

    result = {
        "label": args.label,
        "split": args.split,
        "n_images": len(rows),
        "n_unique_captions": len(unique_captions),
        "recall": recalls,
    }
    print(json.dumps(result, indent=2))

    out_path = Path(args.out) if args.out else PROJECT_ROOT / "results" / f"retrieval_{args.label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
