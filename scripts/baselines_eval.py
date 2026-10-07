"""
Sanity-check baselines for the retrieval numbers in results/retrieval_*.json,
to make clear how much of the reported Recall@K is attributable to caption
class imbalance / chance rather than genuine image-text alignment:

1. Majority-caption baseline (image->text only): a "model" that ignores the
   image entirely and always ranks captions by how frequent they are in the
   split. R@K = fraction of images whose true caption is among the K most
   frequent captions.
2. Caption group-size distribution (text->image): histogram of "how many
   images share each unique caption" in the split -- this determines how
   easy a text->image chance hit is (a caption shared by many images is easy
   to hit by chance; a caption with 1-2 images is hard).
3. Empirical chance baseline: reuse the exact same recall_at_k() code path
   as the real evaluation, but with random unit vectors standing in for both
   image and text embeddings (real model not involved at all). Averaged
   over multiple random seeds to get a stable estimate + std-dev, so the
   real model's numbers can be read as "lift over chance on this exact
   split", not just as raw percentages.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from clip_mvtec.retrieval_eval import recall_at_k

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_rows(captions_csv, split):
    with open(captions_csv, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def build_caption_index(rows):
    unique_captions = sorted({r["caption"] for r in rows})
    caption_to_idx = {c: i for i, c in enumerate(unique_captions)}
    image_caption_idx = torch.tensor([caption_to_idx[r["caption"]] for r in rows])
    return image_caption_idx, unique_captions


def majority_caption_baseline(image_caption_idx, n_captions, ks=(1, 5, 10)):
    group_sizes = torch.bincount(image_caption_idx, minlength=n_captions)
    ranked_captions = torch.argsort(group_sizes, descending=True)  # fixed for every image

    result = {}
    for k in ks:
        top_k = set(ranked_captions[:k].tolist())
        hit = torch.tensor([c in top_k for c in image_caption_idx.tolist()])
        result[k] = hit.float().mean().item()
    return result, group_sizes


def group_size_histogram(group_sizes):
    counts = Counter(group_sizes.tolist())
    return dict(sorted(counts.items()))  # {group_size: n_captions_with_that_size}


def majority_caption_baseline_t2i(image_caption_idx, group_sizes, n_images, ks=(1, 5, 10)):
    """t2i majority baseline: a fixed global ranking of ALL images, ordered by
    the size of their own caption's group (largest group first), used
    identically for every query caption regardless of what that caption is.
    Since caption groups are disjoint, a query caption only ever gets a hit
    when it IS the majority (largest) caption -- every other caption's
    relevant images are ranked strictly below the top `max_group_size`
    positions and can't appear in a top-K with K <= max_group_size."""
    per_image_group_size = group_sizes[image_caption_idx]  # group size of each image's own caption
    # descending by group size, tie-broken by original image index for determinism
    global_ranking = torch.argsort(per_image_group_size, descending=True, stable=True)

    n_captions = group_sizes.shape[0]
    result = {}
    for k in ks:
        top_k_images = set(global_ranking[:k].tolist())
        hits = 0
        for cap_idx in range(n_captions):
            relevant_images = (image_caption_idx == cap_idx).nonzero(as_tuple=True)[0].tolist()
            if top_k_images & set(relevant_images):
                hits += 1
        result[k] = hits / n_captions
    return result


def empirical_random_baseline(image_caption_idx, n_images, n_captions, ks=(1, 5, 10), embed_dim=512, n_trials=30, seed=0):
    i2t_trials = {k: [] for k in ks}
    t2i_trials = {k: [] for k in ks}
    g = torch.Generator().manual_seed(seed)

    for trial in range(n_trials):
        image_embeds = torch.randn(n_images, embed_dim, generator=g)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = torch.randn(n_captions, embed_dim, generator=g)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        recalls = recall_at_k(image_embeds, text_embeds, image_caption_idx, ks=ks)
        for k in ks:
            i2t_trials[k].append(recalls["image_to_text"][k])
            t2i_trials[k].append(recalls["text_to_image"][k])

    def summarize(trials):
        out = {}
        for k in ks:
            vals = torch.tensor(trials[k])
            out[k] = {"mean": vals.mean().item(), "std": vals.std().item()}
        return out

    return {"image_to_text": summarize(i2t_trials), "text_to_image": summarize(t2i_trials)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions-csv", default=str(PROJECT_ROOT / "data" / "captions.csv"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = load_rows(args.captions_csv, args.split)
    image_caption_idx, unique_captions = build_caption_index(rows)
    n_images, n_captions = len(rows), len(unique_captions)

    majority_recall, group_sizes = majority_caption_baseline(image_caption_idx, n_captions)
    majority_recall_t2i = majority_caption_baseline_t2i(image_caption_idx, group_sizes, n_images)
    histogram = group_size_histogram(group_sizes)
    random_baseline = empirical_random_baseline(
        image_caption_idx, n_images, n_captions, n_trials=args.n_trials
    )

    result = {
        "split": args.split,
        "n_images": n_images,
        "n_unique_captions": n_captions,
        "majority_caption_baseline_i2t": majority_recall,
        "majority_caption_baseline_t2i": majority_recall_t2i,
        "caption_group_size_histogram": histogram,  # {images_per_caption: n_captions_with_that_count}
        "group_size_stats": {
            "min": group_sizes.min().item(),
            "max": group_sizes.max().item(),
            "mean": group_sizes.float().mean().item(),
            "median": group_sizes.float().median().item(),
        },
        "empirical_random_baseline": {
            "n_trials": args.n_trials,
            "embed_dim": 512,
            **random_baseline,
        },
    }
    print(json.dumps(result, indent=2))

    out_path = Path(args.out) if args.out else PROJECT_ROOT / "results" / f"baselines_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
