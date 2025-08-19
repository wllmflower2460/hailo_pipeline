#!/bin/bash

# Script to test the Hailo container environment

echo "Testing Hailo development container..."
podman exec -it hailo-dog-training python3 -c "
import torch
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns
import cv2
import sklearn
print(f'Python version: {__import__(\"sys\").version}')
print(f'PyTorch version: {torch.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'OpenCV version: {cv2.__version__}')
print(f'Matplotlib version: {matplotlib.__version__}')
print(f'Pandas version: {pd.__version__}')
print(f'Scikit-learn version: {sklearn.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Using device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')
"
