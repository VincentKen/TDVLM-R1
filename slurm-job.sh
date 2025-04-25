#!/bin/bash
#SBATCH --partition=GPU-a100
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:2

cd /home/e12229949/TDVLM-R1
source /home/e12229949/.bashrc
source /home/e12229949/miniconda3/bin/activate vlm-r1
conda init bash

conda activate vlm-r1

bash setup.sh

pip install -U peft

bash src/open-r1-multimodal/run_scripts/td-dataset.sh
