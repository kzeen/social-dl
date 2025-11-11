#!/bin/bash
set -e

python -m src.train \
  --model logistic \
  --data_dir data \
  --output_dir outputs/logistic
