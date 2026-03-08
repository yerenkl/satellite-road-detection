#!/bin/bash
#BSUB -q gpuv100
#BSUB -W 0:30
#BSUB -J unet
#BSUB -o /work3/s252653/satellite-road-detection-project/jobs/logs/unet_%J.out
#BSUB -e /work3/s252653/satellite-road-detection-project/jobs/logs/unet_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"

echo "Running UNet training..."
uv run python -m src.run model=attention_unet

echo "Training complete! Check the checkpoint directory for outputs."
