# Third-party resources

VECD uses [CLIP ViT-B/16](https://huggingface.co/openai/clip-vit-base-patch16) (OpenAI) and [DINOv2-small](https://huggingface.co/facebook/dinov2-small) (Meta). Backbone weights are loaded through Transformers and remain subject to their original terms.

The benchmarks use EditCLEVR, MagicBrush, HumanEdit, and MaskEdit. Their data and annotations are obtained separately under the respective providers' terms. Python dependencies are listed in `pyproject.toml`; their original licenses apply.
