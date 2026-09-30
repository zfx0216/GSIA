# Table 2

Attack-comparison scripts. Method subfolders contain the GSIA, PGD, PGDRS, EADEN, and AutoAttack implementations and their run scripts. `L2.py`, `PSNR.py`, and `SSIM.py` compare generated adversarial PNGs with their original images and aggregate statistics across three groups.

Configure all `...` paths for images, model checkpoints, class indices, and output folders before running. Run an attack entry point in its method folder, then run the desired metric script from this folder. The metric scripts match images by PNG filename and skip missing pairs.
