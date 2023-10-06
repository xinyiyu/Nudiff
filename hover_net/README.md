# Usage
## Set up environment
Please refer to [hover-net github](https://github.com/vqdang/hover_net).

## Train
1. Download [pretrained ResNet-50 model](https://drive.google.com/file/d/1KntZge40tAHgyXmHYVqZZ5d2p_4Qr2l5/view) to `./pretrained`. The weights of Hover-Net encoder will be initiated using this pretrained model.
2. Split original images of MoNuSeg/Kumar into smaller patches using `extract_patches.py`. Need to specify dataset folder and saving directory.
3. Set training hyperparameters in `./models/hovernet/opt.py` based on the machine and your need (e.g epoches, batch size, checkpoint saving interval).
4. Specify training/validation dataset folders and log directory in `config.py`.
5. Run training with `run_train.py`. Training/validation stats and model checkpoints will be saved in your specified log directory.

## Inference
1. Select one model checkpoint according to validation stats (e.g epoch with max dice).
2. Set parameters in `run_tile.sh`.
3. Run inference with 'run_tile.sh'. Predicted instance maps will be saved in your specified output directory.

## Compute metrics
Compute segmentation metrics (Dice, AJI, PQ) with `compute_stats.py`. You need to specify ground true and prediction directories.


