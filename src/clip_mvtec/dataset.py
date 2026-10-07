import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class CaptionedMVTecDataset(Dataset):
    """Reads data/captions.csv and serves (PIL image, caption) pairs for a
    given `split` ("train" / "val" / "test"). Tokenization/pixel
    preprocessing is left to the collate_fn so the CLIPProcessor can batch
    both modalities at once."""

    def __init__(self, captions_csv: str, project_root: str, split: str):
        self.project_root = Path(project_root)
        self.rows = []
        with open(captions_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No rows found for split={split!r} in {captions_csv}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(self.project_root / row["image_path"]).convert("RGB")
        return {
            "image": image,
            "caption": row["caption"],
            "category": row["category"],
            "label": row["label"],
        }


def make_collate_fn(processor):
    def collate_fn(batch):
        images = [b["image"] for b in batch]
        captions = [b["caption"] for b in batch]
        enc = processor(
            text=captions,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc["captions"] = captions
        enc["categories"] = [b["category"] for b in batch]
        enc["labels_str"] = [b["label"] for b in batch]
        return enc

    return collate_fn
