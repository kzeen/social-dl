#!/bin/bash
set -e

python -m src.evaluate \
  --checkpoint $1 \
  --data_dir data \
  --output_path outputs/predictions.csv
