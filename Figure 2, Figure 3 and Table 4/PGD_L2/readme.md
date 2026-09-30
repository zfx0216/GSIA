# PGD L2 — Figure 2 and Table 4

This folder contains the untargeted projected-gradient L2 attack used for Figure 2 and Table 4. The main run script is `PGD untargeted attack L2.py`; the other Python files provide the attack implementation and helpers.

## Files

`attack.py`, `PGD untargeted attack L2.py`, `PGD.py`

## Run

Replace every `...` in the run script and any imported local module with the correct image, checkpoint, class-index, and output paths. Review the model architecture and L2 budget in the script, then run `PGD untargeted attack L2.py` with Python from this folder. Output images can be evaluated by the metric or success-rate scripts in the parent experiment folder.
