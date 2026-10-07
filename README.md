# CLIP Foundation Model Fine-Tuning + Quantization on MVTec-AD

Fine-tunes `openai/clip-vit-base-patch32` with LoRA on MVTec-AD (industrial
visual defect detection) for image-text retrieval, then quantizes the
fine-tuned model and benchmarks base vs LoRA fine-tuned vs quantized on
Recall@K, latency, and VRAM.

## Environment

Built and run on **native Windows** (not Linux/WSL2 as originally planned —
see [Deviations](#deviations-from-original-plan) below).

- Python 3.11, conda env `clip-lora`
- RTX 4050 Laptop GPU, 6GB VRAM, driver CUDA 12.7
- PyTorch 2.5.1 + CUDA 12.4 (installed via `conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia` — the `pip install --index-url download.pytorch.org` route failed with a TLS handshake error against `download-r2.pytorch.org` from this network; conda's own channels were unaffected)

```bash
conda create -n clip-lora python=3.11 -y
conda activate clip-lora
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

## Deviations from original plan

- **Native Windows instead of Linux/WSL2.** The original spec called for
  Linux/WSL2 since quantization tooling (bitsandbytes, ONNX Runtime) is
  generally better supported there. No WSL2 Ubuntu distro was available on
  this machine at project start, and the user chose to proceed on native
  Windows rather than set one up. Both `bitsandbytes` 8-bit quantization and
  `onnxruntime` installed and ran correctly on this Windows + RTX 4050 setup
  (verified with a real `Linear8bitLt` GPU forward pass before committing to
  this path), so this deviation did not block the project, but a Linux
  environment would likely be more robust for further extension.
- **Dataset source.** MVTec-AD's official site gates the download behind a
  registration form that cannot be automated from here. The Hugging Face
  mirror checked (`Voxel51/mvtec-ad`) only exposes raw images with no
  queryable labels via the plain `datasets` library. Used the Kaggle mirror
  `ipythonx/mvtec-ad` instead, which preserves the original
  `category/{train,test}/{good,defect_type}` folder structure with ground
  truth masks intact.

## Dataset & captions

MVTec-AD ships with **no natural-language captions** — only folder-structure
labels (category, defect type, pixel masks). Captions are generated purely
from those structural labels via fixed templates (`scripts/generate_captions.py`):

- Defective: `"a photo of a {category} with {defect_type}"`
- Normal: `"a {category} showing no visible defects"`

Templates are deliberately coarse (category + defect type only) — they do
**not** encode mask-derived position/size information.

```bash
python scripts/download_mvtec.py
python scripts/generate_captions.py
```

This produces `data/captions.csv` with our own stratified train/val/test
split (grouped by category+label, since MVTec's own "train" folder contains
only defect-free images and is useless alone for a retrieval task that needs
defect-caption diversity in every split).

## Fine-tuning

LoRA adapters (rank 8, alpha 16) on the attention projections
(`q_proj`/`k_proj`/`v_proj`/`out_proj`) of **both** the vision and text
towers, base weights frozen. Standard symmetric CLIP contrastive loss.
Tracked with MLflow.

```bash
python scripts/train_lora.py --epochs 15 --batch-size 32
mlflow ui   # inspect loss curves at localhost:5000
```

## Evaluation

Recall@{1,5,10}, both directions, on the held-out test split:

```bash
python scripts/evaluate_retrieval.py --label base
python scripts/evaluate_retrieval.py --lora-adapter checkpoints/clip-lora-mvtec --label finetuned
```

## Quantization & benchmarking

```bash
python scripts/merge_lora.py
python scripts/benchmark.py --model-dir checkpoints/clip-mvtec-merged --quantization none  --label finetuned
python scripts/benchmark.py --model-dir checkpoints/clip-mvtec-merged --quantization 8bit  --label quantized_8bit
python scripts/make_comparison_table.py
```

`benchmark.py` logs GPU temperature (via `nvidia-smi`) immediately before and
after each timed run, and flags a run as `thermal_throttle_suspected` if the
temperature rose by more than 15C — so latency numbers can be sanity-checked
against thermal state rather than taken at face value.

## Results

Trained 15 epochs, LoRA r=8/alpha=16 on attention projections in both towers
(983K trainable params, 0.65% of the 152M total). Train loss 0.99 -> 0.72;
best val loss 2.868 at epoch 7 (checkpoint used below). Val loss stayed
noisy and did not track train loss closely — expected given only 88 unique
captions and the in-batch caption-collision issue in limitation 3 below.

| Model | R@1 (i2t) | R@5 (i2t) | R@10 (i2t) | R@1 (t2i) | R@5 (t2i) | R@10 (t2i) | Latency (ms/query) | Peak VRAM (MB) | GPU temp after (C) |
|---|---|---|---|---|---|---|---|---|---|
| Base CLIP (zero-shot) | 0.203 | 0.636 | 0.810 | 0.148 | 0.307 | 0.432 | n/a | n/a | n/a |
| LoRA fine-tuned | 0.840 | 0.986 | 1.000 | 0.727 | 0.966 | 1.000 | 1.18 | 641.3 | 54.0 |
| LoRA fine-tuned + 8-bit quantized | 0.849 | 0.989 | 1.000 | 0.693 | 0.955 | 1.000 | 6.39 | 217.8 | 54.0 |

(Also in [`results/comparison_table.md`](results/comparison_table.md); raw
per-run numbers in `results/retrieval_*.json` and `results/benchmark_*.json`.)

**Fine-tuning lift** is large: R@1 image->text roughly quadruples (20.3% ->
84.0%), R@1 text->image roughly quintuples (14.8% -> 72.7%), R@10 saturates
at 100% in both directions. Read this as expected, not impressive on its
own merits: the retrieval task is closed-vocabulary (only 88 unique
captions total), so this is much closer to "learn to classify 88 fixed
category+defect labels from images" than open-vocabulary retrieval — see
limitation 2 below.

**Baselines, to calibrate how much of that lift is real** (`scripts/baselines_eval.py`,
full output in `results/baselines_test.json`):

| | R@1 (i2t) | R@5 (i2t) | R@10 (i2t) |
|---|---|---|---|
| Pure chance (empirical, 30 trials of random embeddings through the real eval code) | 1.25% ± 0.31% | 5.73% ± 0.86% | 11.58% ± 0.99% |
| Majority-caption baseline (ignore the image, always rank by caption frequency) | 8.06% | 31.6% | 57.4% |
| LoRA fine-tuned (from table above) | 84.0% | 98.6% | 100% |

The majority-caption baseline is well above uniform chance (8.06% vs. the
1/88 = 1.14% you'd expect from 88 equiprobable captions) because the caption
distribution is heavily imbalanced: of 88 unique captions in the test split,
15 cover only 1 image each while the single largest caption (almost
certainly a "good"/no-defect caption for a common category) covers 64
images (group-size distribution: min 1, median 3, mean 9.0, max 64). Text->image
chance is governed by the same imbalance — a caption shared by 64 images is
an easy chance-hit, one shared by 1 image is not.

Measured against the honest floor (chance) rather than zero-shot CLIP, the
fine-tuned model's R@1 lift is +82.8pp over pure chance and +75.9pp over the
majority-caption baseline — the earlier "large lift" claim holds up even
under a strict floor, not just relative to zero-shot.

**Stratified by caption group-size** (`scripts/stratified_group_size_eval.py`,
full output in `results/stratified_group_size_eval.json`), i2t only, real
model vs. the same empirical chance baseline restricted to the same query
subset — this checks whether the blended 84.0% R@1 average above is riding
almost entirely on the easy, well-represented captions:

| Stratum | n images | R@1 chance | R@1 real | R@5 chance | R@5 real | R@10 chance | R@10 real |
|---|---|---|---|---|---|---|---|
| Rare (group-size 1-2, 38 captions) | 61 | 1.80% ± 2.17% | 55.74% | 6.17% ± 2.85% | 98.36% | 11.80% ± 4.09% | 100% |
| Common (group-size 10+, 15 captions) | 616 | 1.24% ± 0.37% | 93.02% | 5.81% ± 1.03% | 98.70% | 11.73% ± 1.14% | 100% |

The model's edge over chance holds clearly on "rare" captions, not just
"common" ones (+54pp vs +92pp at R@1) — the headline number is not solely
an artifact of the easy long-tail cases. That said, the strata are not
equivalent: R@1 on rare captions (55.74%) is meaningfully below R@1 on
common captions (93.02%), so there is a real, honest generalization gap
between well-represented and singleton/near-singleton defect types that the
blended all-test-images average (84.0%) obscures. R@5 and R@10 saturate to
~98-100% in both strata regardless, so this gap is specific to the top-1
retrieval case.

Same baselines, text->image direction. The t2i majority baseline here is a
fixed global ranking of all 794 test images ordered by their own caption's
group size (largest group first) — used identically for every query
caption. Because caption groups are disjoint, that fixed ranking only ever
produces a hit when the query caption IS the majority (largest) one, so the
result is flat across K:

| | R@1 (t2i) | R@5 (t2i) | R@10 (t2i) |
|---|---|---|---|
| Pure chance (empirical, 30 trials) | 1.25% ± 1.09% | 6.17% ± 2.36% | 10.08% ± 2.94% |
| Majority-caption baseline (fixed ranking by group size, ignores the query caption) | 1.14% | 1.14% | 1.14% |
| LoRA fine-tuned (from table above) | 72.7% | 96.6% | 100% |

The t2i majority baseline (1.14% = 1/88 flat) is actually *weaker* than pure
chance here, unlike the i2t majority baseline (8.06%, well above its 1.14%
chance floor). The asymmetry is structural: "always guess the most frequent
caption" helps on the i2t side because many different query images
genuinely share that one frequent caption, so a large fraction of image
queries can match it. "Always guess the images from the most frequent
caption-group" does not have the mirror benefit on the t2i side, because
caption groups are mutually exclusive — those fixed images only ever satisfy
the one query caption they actually belong to, and the other 87 query
captions get zero credit no matter what K is. The fine-tuned model's t2i R@1
lift is +71.5pp over pure chance and +71.6pp over the majority baseline.

**Quantization tradeoff is not what naive intuition predicts.** 8-bit
(bitsandbytes `Linear8bitLt`) cut peak VRAM by ~66% (641MB -> 218MB) but made
inference **~5.4x slower** (1.18ms -> 6.39ms per query), not faster.
This is a real, reproducible property of bitsandbytes on this hardware/model
combination, not a bug: its int8 kernels carry a fixed per-call
quantize/dequantize overhead that dominates when the underlying matrices are
already small (CLIP-B/32 is ~150M params), so there's no throughput win to
offset it on this GPU. Recall@K moved by only 1-3pp in either direction
(noise-level), consistent with 8-bit quantization's usual near-lossless
accuracy behavior. GPU temperature was stable (49-54C) across both
benchmark runs with no thermal throttling detected.

## Honest limitations

1. **Captions are templated/synthetic, not human-authored.** They encode
   only category and defect type, generated mechanically from MVTec-AD's
   folder structure. This is not a substitute for a real captioning dataset
   and the retrieval task is correspondingly easier than one built on
   free-form natural language.
2. **Partial label leakage.** Because captions are derived from the same
   categorical labels used to define ground-truth matches, retrieval can be
   partially solved by category/defect-type classification rather than
   genuine fine-grained image-text alignment. Templates were kept
   deliberately coarse (no position/size/mask-derived detail) specifically
   to limit this, but it is a **mitigation, not a fix** — a model that
   learns "bottle photos with `contamination` in the caption look like this"
   is still doing something closer to classification than open-vocabulary
   retrieval.
3. **Caption collisions in-batch.** Many images share an identical caption
   string (same category+label). The contrastive loss used here treats only
   the diagonal of each batch as the positive pair, so any other image in
   the batch that happens to share the same caption is incorrectly treated
   as a hard negative. This is a known simplification, not fixed in v1.
4. **Small per-defect-type sample sizes.** Some MVTec-AD defect subtypes
   have only a handful of images; Recall@K on those slices is noisy and
   should not be over-interpreted category-by-category.
5. Thermal state during benchmarking is reported, not assumed benign — see
   `gpu_temp_c_before/after` and `thermal_throttle_suspected` in
   `results/benchmark_*.json`.
6. Results reflect a single training run with no seed variance reported;
   Recall@K point estimates, particularly on the low-sample rare stratum
   (n=61 images), should be read with that in mind.
