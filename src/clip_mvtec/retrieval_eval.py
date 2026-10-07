"""Shared Recall@K logic for image<->text retrieval on the MVTec caption set.

Because captions are templated from (category, label), many images share
the *identical* caption string. So:
  - image -> text: each image has exactly one correct caption (deterministic
    function of its category+label) -> standard single-relevant-item Recall@K.
  - text -> image: each unique caption may have many relevant images (every
    image with that category+label) -> Recall@K = at least one relevant
    image appears in the top-K.
This asymmetry is a direct consequence of template-based captioning and is
called out in the README limitations.
"""
import torch


@torch.no_grad()
def embed_all(model, processor, rows, project_root, device, batch_size=32):
    from PIL import Image

    image_embeds, text_embeds_by_caption = [], {}
    unique_captions = sorted({r["caption"] for r in rows})
    caption_to_idx = {c: i for i, c in enumerate(unique_captions)}

    # Text embeddings (once per unique caption).
    for i in range(0, len(unique_captions), batch_size):
        chunk = unique_captions[i:i + batch_size]
        enc = processor(text=chunk, return_tensors="pt", padding=True, truncation=True).to(device)
        feats = model.get_text_features(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        feats = feats / feats.norm(dim=-1, keepdim=True)
        for j, c in enumerate(chunk):
            text_embeds_by_caption[c] = feats[j].cpu()

    text_embeds = torch.stack([text_embeds_by_caption[c] for c in unique_captions])

    # Image embeddings.
    image_caption_idx = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        images = [Image.open(project_root / r["image_path"]).convert("RGB") for r in chunk]
        enc = processor(images=images, return_tensors="pt").to(device)
        feats = model.get_image_features(pixel_values=enc["pixel_values"])
        feats = feats / feats.norm(dim=-1, keepdim=True)
        image_embeds.append(feats.cpu())
        image_caption_idx.extend(caption_to_idx[r["caption"]] for r in chunk)

    image_embeds = torch.cat(image_embeds, dim=0)
    image_caption_idx = torch.tensor(image_caption_idx)
    return image_embeds, text_embeds, image_caption_idx, unique_captions


def image_to_text_hits(image_embeds, text_embeds, image_caption_idx, ks=(1, 5, 10)):
    """Per-image i2t hit indicator (not averaged), so callers can subset by
    stratum (e.g. by caption group-size) before computing a mean Recall@K
    instead of only being able to compute it over the whole query set."""
    sim = image_embeds @ text_embeds.t()
    ranked = sim.argsort(dim=1, descending=True)
    hits = {}
    for k in ks:
        topk = ranked[:, :k]
        hits[k] = (topk == image_caption_idx.unsqueeze(1)).any(dim=1)
    return hits


def recall_at_k(image_embeds, text_embeds, image_caption_idx, ks=(1, 5, 10)):
    sim = image_embeds @ text_embeds.t()  # [n_images, n_captions]

    # image -> text
    i2t = {}
    ranked = sim.argsort(dim=1, descending=True)
    for k in ks:
        topk = ranked[:, :k]
        hit = (topk == image_caption_idx.unsqueeze(1)).any(dim=1)
        i2t[k] = hit.float().mean().item()

    # text -> image: relevant set = all images whose caption == this caption
    n_captions = text_embeds.shape[0]
    relevant = [[] for _ in range(n_captions)]
    for img_idx, cap_idx in enumerate(image_caption_idx.tolist()):
        relevant[cap_idx].append(img_idx)

    sim_t2i = sim.t()  # [n_captions, n_images]
    ranked_t2i = sim_t2i.argsort(dim=1, descending=True)
    t2i = {}
    for k in ks:
        hits = 0
        for cap_idx in range(n_captions):
            topk = set(ranked_t2i[cap_idx, :k].tolist())
            if topk & set(relevant[cap_idx]):
                hits += 1
        t2i[k] = hits / n_captions

    return {"image_to_text": i2t, "text_to_image": t2i}
