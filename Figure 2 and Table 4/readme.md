# Figure 2 and Table 4

`mobilenetv2_train.py` and `resnet18_train.py` train or fine-tune image classifiers and save checkpoints, logs, and learning curves. The method subfolders contain GSIA, PGD, and AutoAttack experiments against the resulting models.

Configure `...` paths in each training script for the training and test `ImageFolder` datasets, pretrained weights, and checkpoint output. Then configure the attack scripts with the intended trained checkpoint, class-index JSON, test images, and attack-output directory. Verify the class count and checkpoint architecture before running.
