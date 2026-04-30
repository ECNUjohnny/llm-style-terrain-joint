# AGENTS.md — llm-style-terrain-joint

## Project status

This is an early scaffold. Most code is skeleton with `NotImplementedError` / `TODO` markers. The only runnable component is `HeightMapVAE` (inherits `diffusers.AutoencoderKL`).

## Entry points

```bash
python main.py --mode train       # training scaffolding (TODO)
python main.py --mode inference   # inference scaffolding (TODO)
python scripts/height_vae/train_height_vae.py --mode train   # only real training script
python scripts/height_vae/train_height_vae.py --mode test --checkpoint <path>
```

## Architecture

- **Joint latent**: `torch.cat([height_latent, texture_latent], dim=1)` → `[B, 8, 64, 64]`
  - channels 0-3: height latent (custom VAE)
  - channels 4-7: texture latent (SD VAE from `stabilityai/stable-diffusion-2-1`)
- **HeightMapVAE** (`models/vae/heightmap_vae.py`): modifies `diffusers.AutoencoderKL` with `in_channels=1, out_channels=1, latent_channels=4, sample_size=512`
  - Input/output normalization: `x / 3000.0` ↔ `x * 3000.0`
  - Custom geo loss: `slope_loss + 0.5 * curvature_loss` (weighted 0.8 in total VAE loss)
- **UNet** (`models/unet/unet_8ch.py`): 8-in 8-out, expects `CrossAttnDownBlock2D`/`CrossAttnUpBlock2D` blocks — placeholder only
- **Text encoder** (`models/clip/text_encoder.py`): dual-branch CLIP (`openai/clip-vit-base-patch32`) — placeholder only
- **DDIM**: 50 steps, guidance_scale=7.5 — scheduler in `utils/latent_utils.py` (placeholder)
- Config is **hardcoded** in `main.py` — no config files or CLI overrides

## Package management

Uses **uv** (not pip). Lockfile: `uv.lock`. Python 3.11 required.

```bash
uv sync        # install dependencies
uv run python  ...  # or `.venv/bin/python ...`
```

## Important gotchas

- `dataset/height_map_dataset.py` is a stub (contains only `1`). There is no working `HeightMapDataset` class despite imports in `scripts/height_vae/train_height_vae.py`.
- `data/` gitignore ignores everything (`*` at root). Raw data goes in `data/origin/`.
- `scripts/data_process/` is empty — no data pipeline exists yet.
- No linter, formatter, type checker, pre-commit hooks, or CI configured.
- Texture VAE and CLIP are placeholders (`texture_vae = None`, `text_encoder` raises `NotImplementedError`).
- The UNet uses a `placeholder_conv` pass-through — no actual diffusion backbone.
- All docstrings are in Chinese; keep comments Chinese when modifying existing Chinese files.