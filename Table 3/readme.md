# Table 3

Perturbation-budget experiments using GSIA, PGD, PGDRS, EADEN, and AutoAttack. `budget_l2.py` supplies the ImageNet-normalized L2 projection helpers used by these attacks. `testify.py` compares predictions on original and adversarial images and reports attack success.

Set the `...` paths in the attack scripts and in `testify.py` to the corresponding images, weights, class-index JSON, and output directories. Run an attack first, then evaluate its saved images with `testify.py`. Check the `BUDGET` constant and per-script settings when reproducing a specific row.
