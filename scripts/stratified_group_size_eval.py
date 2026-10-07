"""
Stratified i2t Recall@K by caption group-size, real fine-tuned model vs.
empirical random-chance baseline, on the same query subsets.

Strata:
  rare   -- captions with group-size 1-2 (singleton/near-singleton defect
            types: hardest cases, least data)
  common -- captions with group-size >= 10 (the long tail, dominated by
            "good"/no-defect captions for well-represented categories)

This exists to check whether the fine-tuned model's headline Recall@K is a
genuine gain across the board, or whether it's riding almost entirely on the
easy, well-represented "common" captions while barely beating chance on
"rare" ones -- the blended average over all 794 test images can hide that
distinction.
"""
import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from clip_mvtec.model import BASE_MODEL_ID, BASE_MODEL_REVISION
from clip_mvtec.retrieval_eval import embed_all, image_to_text_hits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KS = (1, 5, 10)


def load_rows(captions_csv, split):
    with open(captions_csv, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def stratum_masks(image_caption_idx, n_captions):
    group_sizes = torch.bincount(image_caption_idx, minlength=n_captions)
    rare_captions = (group_sizes >= 1) & (group_sizes <= 2)
    common_captions = group_sizes >= 10

    per_image_group_size = group_sizes[image_caption_idx]
    rare_mask = rare_captions[image_caption_idx]
    common_mask = common_captions[image_caption_idx]

    return {
        "rare (group-size 1-2)": rare_mask,
        "common (group-size 10+)": common_mask,
    }, group_sizes, rare_captions, common_captions


def real_model_recall_by_stratum(model, processor, rows, device, masks):
    image_embeds, text_embeds, image_caption_idx, unique_captions = embed_all(
        model, processor, rows, PROJECT_ROOT, device
    )
    hits = image_to_text_hits(image_embeds, text_embeds, image_caption_idx, ks=KS)

    result = {}
    for name, mask in masks.items():
        result[name] = {k: hits[k][mask].float().mean().item() for k in KS}
        result[name]["n_images"] = int(mask.sum().item())
    return result, image_caption_idx


def chance_recall_by_stratum(image_caption_idx, n_captions, masks, n_trials=30, embed_dim=512, seed=0):
    n_images = image_caption_idx.shape[0]
    trials = {name: {k: [] for k in KS} for name in masks}
    g = torch.Generator().manual_seed(seed)

    for _ in range(n_trials):
        image_embeds = torch.randn(n_images, embed_dim, generator=g)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = torch.randn(n_captions, embed_dim, generator=g)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        hits = image_to_text_hits(image_embeds, text_embeds, image_caption_idx, ks=KS)
        for name, mask in masks.items():
            for k in KS:
                trials[name][k].append(hits[k][mask].float().mean().item())

    result = {}
    for name in masks:
        result[name] = {}
        for k in KS:
            vals = torch.tensor(trials[name][k])
            result[name][k] = {"mean": vals.mean().item(), "std": vals.std().item()}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions-csv", default=str(PROJECT_ROOT / "data" / "captions.csv"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--lora-adapter", default=str(PROJECT_ROOT / "checkpoints" / "clip-lora-mvtec"))
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_rows(args.captions_csv, args.split)

    from transformers import CLIPModel, CLIPProcessor
    from peft import PeftModel

    model = CLIPModel.from_pretrained(BASE_MODEL_ID, revision=BASE_MODEL_REVISION).to(device)
    processor = CLIPProcessor.from_pretrained(BASE_MODEL_ID, revision=BASE_MODEL_REVISION)
    model = PeftModel.from_pretrained(model, args.lora_adapter)
    model = model.merge_and_unload()
    model.eval()

    unique_captions = sorted({r["caption"] for r in rows})
    caption_to_idx = {c: i for i, c in enumerate(unique_captions)}
    image_caption_idx_precheck = torch.tensor([caption_to_idx[r["caption"]] for r in rows])
    n_captions = len(unique_captions)

    masks, group_sizes, rare_captions, common_captions = stratum_masks(image_caption_idx_precheck, n_captions)
    print(f"rare captions: {int(rare_captions.sum())}, common captions: {int(common_captions.sum())}")

    real_result, image_caption_idx = real_model_recall_by_stratum(model, processor, rows, device, masks)
    assert torch.equal(image_caption_idx, image_caption_idx_precheck), "caption indexing mismatch"

    chance_result = chance_recall_by_stratum(image_caption_idx, n_captions, masks, n_trials=args.n_trials)

    print(f"\n{'Stratum':<25} {'n_images':>9}  " + "  ".join(f"R@{k} chance / real" for k in KS))
    for name in masks:
        n_img = real_result[name]["n_images"]
        cells = []
        for k in KS:
            c = chance_result[name][k]
            r = real_result[name][k]
            cells.append(f"{c['mean']*100:5.2f}%±{c['std']*100:4.2f} / {r*100:5.2f}%")
        print(f"{name:<25} {n_img:>9}  " + "  ".join(cells))

    import json
    out = {
        "split": args.split,
        "rare_captions": int(rare_captions.sum()),
        "common_captions": int(common_captions.sum()),
        "real_model": real_result,
        "chance": chance_result,
    }
    out_path = PROJECT_ROOT / "results" / "stratified_group_size_eval.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
