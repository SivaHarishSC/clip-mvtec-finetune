import torch
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel, CLIPProcessor

BASE_MODEL_ID = "openai/clip-vit-base-patch32"

# The main-branch revision of this repo predates safetensors and only ships
# pytorch_model.bin, which transformers >=4.50 refuses to torch.load on
# torch<2.6 (CVE-2025-32434). refs/pr/66 is HF's own bot-opened PR adding a
# model.safetensors conversion of the identical weights -- pinned by commit
# sha for reproducibility.
BASE_MODEL_REVISION = "c237dc49a33fc61debc9276459120b7eac67e7ef"

# Attention projections in both the vision and text transformer towers.
# CLIPModel exposes these as `...self_attn.{q,k,v,out}_proj` regardless of
# whether they belong to vision_model or text_model, so a plain suffix
# match picks up both towers.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]


def load_base_model_and_processor(device: str = "cuda"):
    model = CLIPModel.from_pretrained(BASE_MODEL_ID, revision=BASE_MODEL_REVISION)
    processor = CLIPProcessor.from_pretrained(BASE_MODEL_ID, revision=BASE_MODEL_REVISION)
    model.to(device)
    return model, processor


def apply_lora(model: CLIPModel, r: int = 8, alpha: int = 16, dropout: float = 0.1):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    return model


def clip_contrastive_loss(logits_per_image: torch.Tensor) -> torch.Tensor:
    """Standard symmetric CLIP loss. Assumes the diagonal is the only
    positive pair in the batch (see README limitations re: caption reuse
    across images with identical category+defect label)."""
    batch_size = logits_per_image.shape[0]
    targets = torch.arange(batch_size, device=logits_per_image.device)
    loss_i2t = torch.nn.functional.cross_entropy(logits_per_image, targets)
    loss_t2i = torch.nn.functional.cross_entropy(logits_per_image.t(), targets)
    return (loss_i2t + loss_t2i) / 2
