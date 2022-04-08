
# CSCI6633 - Group 3
**ISBI 2012 Analysis of U-Net Architectures**

## Dependencies (Tested with Python 3.10)
- tensorflow
- scipy
- matplotlib
- pandas
- numpy

## Execute scripts (One for each model-type)
- run_unet.py
- run_ternaus.py
- run_pix2pix.py
- run_multiresunet.py
- run_dcunet.py

Each runner will build that U-Net architecture, train it, and test it, and save the results into a results folder.

## Training Experiments
- 1 iteration / 10 Epochs / 500 training steps, 100 validation steps @ 512x512 (U-Net Baseline only!)
- 5 iterations / 20 Epochs / 250 training steps, 100 validation steps @ 256x256
- 20 iterations / 1 Epoch / 50 training steps, 100 validation steps @ 256x256
