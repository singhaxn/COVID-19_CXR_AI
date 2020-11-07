#!/bin/bash

# Run this as root

apt-get update
apt-get install build-essential python3 python3-pip libsm6 libxext6 libxrender-dev imagemagick libgl1-mesa-glx wget -y
pip3 install --upgrade pip
