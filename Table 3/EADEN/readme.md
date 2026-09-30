# EADEN — Table 2

This folder contains the untargeted elastic-net attack used for Table 2. The main run script is `EADEN untargeted attack L2.py`; the other Python files provide the attack implementation and helpers. For Table 2, the attack may import the shared `../budget_l2.py` projection helpers.

## Files

`attack.py`, `EADEN untargeted attack L2.py`, `EADEN.py`

## Run

Replace every `...` in the run script and any imported local module with the correct image, checkpoint, class-index, and output paths. Review the model architecture and L2 budget in the script, then run `EADEN untargeted attack L2.py` with Python from this folder. Output images can be evaluated by the metric or success-rate scripts in the parent experiment folder.
