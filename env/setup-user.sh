#!/bin/bash

# Run this as user

# Torch - CUDA
# pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# Torch - CPU
pip3 install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Other packages
pip3 install --user -r requirements.txt

./download-models.sh
