"""Aggregate results/retrieval_*.json and results/benchmark_*.json into one
markdown comparison table: base vs LoRA fine-tuned vs quantized."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def main():
    base = load_json(RESULTS_DIR / "retrieval_base.json")
    finetuned = load_json(RESULTS_DIR / "retrieval_finetuned.json")
    bench_finetuned = load_json(RESULTS_DIR / "benchmark_finetuned.json")
    bench_quantized = load_json(RESULTS_DIR / "benchmark_quantized_8bit.json")

    rows = []
    if base:
        r = base["recall"]
        rows.append(("Base CLIP (zero-shot)", r["image_to_text"], r["text_to_image"], None, None, None))
    if finetuned:
        r = finetuned["recall"]
        lat = bench_finetuned["latency_ms_per_query"] if bench_finetuned else None
        vram = bench_finetuned["peak_vram_mb"] if bench_finetuned else None
        temp = bench_finetuned.get("gpu_temp_c_after") if bench_finetuned else None
        rows.append(("LoRA fine-tuned", r["image_to_text"], r["text_to_image"], lat, vram, temp))
    if bench_quantized:
        r = bench_quantized["recall"]
        rows.append((
            "LoRA fine-tuned + 8-bit quantized",
            r["image_to_text"], r["text_to_image"],
            bench_quantized["latency_ms_per_query"],
            bench_quantized["peak_vram_mb"],
            bench_quantized.get("gpu_temp_c_after"),
        ))

    lines = [
        "| Model | R@1 (i2t) | R@5 (i2t) | R@10 (i2t) | R@1 (t2i) | R@5 (t2i) | R@10 (t2i) | Latency (ms/query) | Peak VRAM (MB) | GPU temp after (C) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, i2t, t2i, lat, vram, temp in rows:
        lines.append(
            f"| {name} | {fmt(i2t['1'])} | {fmt(i2t['5'])} | {fmt(i2t['10'])} | "
            f"{fmt(t2i['1'])} | {fmt(t2i['5'])} | {fmt(t2i['10'])} | "
            f"{fmt(lat, 2) if lat else 'n/a'} | {fmt(vram, 1) if vram else 'n/a'} | "
            f"{fmt(temp, 1) if temp else 'n/a'} |"
        )

    table = "\n".join(lines)
    print(table)
    out_path = RESULTS_DIR / "comparison_table.md"
    out_path.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
