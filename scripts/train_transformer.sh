#!/bin/bash
set -e

python -m src.train \
  --model bert \
  --data_dir data \
  --output_dir outputs/bert \
  --epochs 3 \
  --batch_size 16
