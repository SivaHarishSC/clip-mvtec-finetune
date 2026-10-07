"""
Download and extract the MVTec-AD dataset via the Kaggle mirror
(ipythonx/mvtec-ad), which preserves the original folder layout:

    data/raw/<category>/train/good/*.png
    data/raw/<category>/test/good/*.png
    data/raw/<category>/test/<defect_type>/*.png
    data/raw/<category>/ground_truth/<defect_type>/*_mask.png

Requires a Kaggle API token. Point KAGGLE_CONFIG_DIR at the directory
containing kaggle.json (the file itself must be named kaggle.json).
"""
import argparse
import os
import zipfile
from pathlib import Path

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-config-dir", default=r"E:\foundational_model",
                         help="Directory containing kaggle.json")
    parser.add_argument("--dataset", default="ipythonx/mvtec-ad",
                         help="Kaggle dataset slug")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "data"))
    args = parser.parse_args()

    os.environ["KAGGLE_CONFIG_DIR"] = args.kaggle_config_dir

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    raw_dir = Path(args.out_dir) / "raw"
    zip_dir = Path(args.out_dir)
    zip_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.dataset} from Kaggle -> {zip_dir}")
    api.dataset_download_files(args.dataset, path=str(zip_dir), unzip=False, quiet=False)

    zip_path = zip_dir / (args.dataset.split("/")[-1] + ".zip")
    if not zip_path.exists():
        candidates = list(zip_dir.glob("*.zip"))
        if not candidates:
            raise FileNotFoundError("No zip file downloaded from Kaggle.")
        zip_path = candidates[0]

    print(f"Extracting {zip_path} -> {raw_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)

    found = [c for c in CATEGORIES if any(raw_dir.rglob(c))]
    print(f"Categories found under {raw_dir}: {len(found)}/{len(CATEGORIES)}")
    print("Done.")


if __name__ == "__main__":
    main()
