"""Merge a trained LoRA adapter into the base CLIP weights and save a
standalone full-precision checkpoint. Quantization (bitsandbytes 8-bit) is
applied at *load* time in benchmark.py, not baked into a saved file, so this
merged checkpoint is the single source both the fine-tuned and quantized
benchmark configs load from."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from clip_mvtec.model import BASE_MODEL_ID, BASE_MODEL_REVISION

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-adapter", default=str(PROJECT_ROOT / "checkpoints" / "clip-lora-mvtec"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "checkpoints" / "clip-mvtec-merged"))
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(BASE_MODEL_ID, revision=BASE_MODEL_REVISION)
    model = PeftModel.from_pretrained(model, args.lora_adapter)
    model = model.merge_and_unload()

    processor = CLIPProcessor.from_pretrained(BASE_MODEL_ID, revision=BASE_MODEL_REVISION)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)
    print(f"Merged model saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
