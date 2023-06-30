#!/bin/bash

#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=1:00:00
#PJM -L jobenv=singularity
#PJM -j

eval "$(/home/z40351r/anaconda3/bin/conda shell.bash hook)"
conda activate vap2

exp_dir=exp/callhome_eng
audio_name=jpn_sample_cut.wav 

python3 run.py \
  --audio example/${audio_name} \
  --exp_dir ${exp_dir} \
  -sd ${exp_dir}/best_model.pt\
  --plot \
