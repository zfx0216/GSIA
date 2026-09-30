# Table 1

GSIA ablation scripts: `GSIA.py` is the standard mask variant, `GSIA_fixed.py` uses a fixed mask, and `GSIA_none.py` removes the mask. `L2.py`, `PSNR.py`, and `SSIM.py` measure perturbation size and image quality for generated PNGs.

Configure every `...` path for input images, model weights, class indices, and output images. Run an ablation variant and then the metric scripts against its output. Keep image filenames aligned with the originals.
