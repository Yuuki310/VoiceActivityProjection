#!/bin/bash

#PJM -L rscgrp=cx-single
#PJM -L gpu=4
#PJM -L elapse=10:00:00
#PJM -L jobenv=singularity
#PJM -j

eval "$(/home/z40351r/anaconda3/bin/conda shell.bash hook)"
conda activate vap2

exp_dir=exp/callhome_eng
python3 vap/train.py \
 --data_train_path None \
 --data_val_path None \
 --exp_dir ${exp_dir} \
 --data_conf_path ${exp_dir}/conf/dset_conf.yaml \
