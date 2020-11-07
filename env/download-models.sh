#!/bin/bash

# Download latest models from onedrive

DROPBOX_URL=https://www.dropbox.com/s/qawazzv0b9je8x0/20201105.tar.xz?dl=1

mkdir -p ../models
cd ../models
wget -c "$DROPBOX_URL" -O current.tar.xz

tar -xf current.tar.xz
