"""
Generate templated image-text captions from the MVTec-AD folder structure.

MVTec-AD ships with no natural-language captions. This script derives them
purely from structural labels already present in the folder layout:

    data/raw/<category>/train/good/*.png
    data/raw/<category>/test/good/*.png
    data/raw/<category>/test/<defect_type>/*.png

Templates (deliberately coarse -- category + defect type only, no
mask-derived position/size details, to limit label leakage into the
retrieval task):

    defective: "a photo of a {category} with {defect_type}"
    normal:    "a {category} showing no visible defects"

Output: data/captions.csv with columns
    image_path, category, label, split_orig, caption, split

`split` (train/val/test) is our own stratified re-split over the union of
MVTec's train+test images, grouped by (category, label), so that every
defect type present in the data appears in more than one of our splits
whenever it has enough examples. This is necessary because MVTec's own
"train" folder contains only "good" images -- useless on its own for a
retrieval task that needs defect-caption diversity in every split.
"""
import argparse
import csv
import random
from pathlib import Path

SEED = 42


def humanize(token: str) -> str:
    return token.replace("_", " ")


def build_caption(category: str, label: str) -> str:
    cat = humanize(category)
    if label == "good":
        return f"a {cat} showing no visible defects"
    return f"a photo of a {cat} with {humanize(label)}"


def collect_images(raw_dir: Path):
    """Walk data/raw/<category>/{train,test}/<label>/*.png|jpg"""
    records = []
    for category_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        category = category_dir.name
        for split_orig in ("train", "test"):
            split_dir = category_dir / split_orig
            if not split_dir.exists():
                continue
            for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                label = label_dir.name  # "good" or a defect type
                for img_path in sorted(label_dir.glob("*.png")) + sorted(label_dir.glob("*.jpg")):
                    records.append({
                        "image_path": str(img_path.relative_to(raw_dir.parent.parent)),
                        "category": category,
                        "label": label,
                        "split_orig": split_orig,
                    })
    return records


def stratified_split(records, train_frac=0.7, val_frac=0.15):
    """Deterministic per-(category,label) group split so every group is
    represented in train, and in val/test whenever it has enough examples."""
    rng = random.Random(SEED)
    groups = {}
    for r in records:
        groups.setdefault((r["category"], r["label"]), []).append(r)

    for key, items in groups.items():
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, round(n * train_frac))
        n_val = max(1 if n - n_train >= 2 else 0, round(n * val_frac))
        n_val = min(n_val, n - n_train)
        for i, r in enumerate(items):
            if i < n_train:
                r["split"] = "train"
            elif i < n_train + n_val:
                r["split"] = "val"
            else:
                r["split"] = "test"
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "raw"))
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "captions.csv"))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    records = collect_images(raw_dir)
    if not records:
        raise SystemExit(f"No images found under {raw_dir}. Did the download/extract succeed?")

    for r in records:
        r["caption"] = build_caption(r["category"], r["label"])

    records = stratified_split(records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "category", "label", "split_orig", "caption", "split"])
        writer.writeheader()
        writer.writerows(records)

    n_by_split = {}
    for r in records:
        n_by_split[r["split"]] = n_by_split.get(r["split"], 0) + 1
    n_categories = len({r["category"] for r in records})
    n_defect_types = len({r["label"] for r in records if r["label"] != "good"})

    print(f"Wrote {len(records)} captioned rows -> {out_path}")
    print(f"Categories: {n_categories}, distinct defect types: {n_defect_types}")
    print("Split sizes:", n_by_split)


if __name__ == "__main__":
    main()
