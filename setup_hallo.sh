#!/bin/bash

cd /workspace
git clone https://github.com/fudan-generative-vision/hallo2.git
cd hallo2

python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install diffusers transformers accelerate opencv-python imageio imageio-ffmpeg "huggingface_hub[cli]"

huggingface-cli download fudan-generative-ai/hallo2 --local-dir ./pretrained_models
