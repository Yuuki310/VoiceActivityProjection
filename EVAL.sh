#!/bin/bash

#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=1:00:00
#PJM -L jobenv=singularity
#PJM -j

eval "$(/home/z40351r/anaconda3/bin/conda shell.bash hook)"
conda activate vap2

exp_dir=exp/sample
python3 vap/evaluation.py \
    --seed 1 \
    --exp_dir ${exp_dir}\
    --state_dict ${exp_dir}/best_model.pt \
    --test_dataset callhome_jpn
