| Model | R@1 (i2t) | R@5 (i2t) | R@10 (i2t) | R@1 (t2i) | R@5 (t2i) | R@10 (t2i) | Latency (ms/query) | Peak VRAM (MB) | GPU temp after (C) |
|---|---|---|---|---|---|---|---|---|---|
| Base CLIP (zero-shot) | 0.203 | 0.636 | 0.810 | 0.148 | 0.307 | 0.432 | n/a | n/a | n/a |
| LoRA fine-tuned | 0.840 | 0.986 | 1.000 | 0.727 | 0.966 | 1.000 | 1.18 | 641.3 | 54.0 |
| LoRA fine-tuned + 8-bit quantized | 0.849 | 0.989 | 1.000 | 0.693 | 0.955 | 1.000 | 6.39 | 217.8 | 54.0 |
