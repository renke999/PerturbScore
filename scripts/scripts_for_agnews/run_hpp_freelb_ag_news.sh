#!/usr/bin/env bash

function runexp {
gpu=${1}      # The GPU you want to use
alr=${2}      # Step size of gradient ascent
amag=${3}     # Magnitude of initial (adversarial?) perturbation
anorm=${4}    # Maximum norm of adversarial perturbation
asteps=${5}   # Number of gradient ascent steps for the adversary

root_path=tensors/
file_name=freelb_freelb_ag_news_alr${alr}_amag${amag}_anm${anorm}_as${asteps}
save_path=${root_path}${file_name}.pt
output_path=hpp_outputs/${file_name}

CUDA_VISIBLE_DEVICES=${gpu} python ../model/freelb.py \
  --model_name_or_path {{YOUR_MODEL_PATH}} \
  --dataset_name ag_news \
  --do_train \
  --max_seq_length 128 \
  --pad_to_max_length True \
  --per_device_train_batch_size 128 \
  --output_dir ${output_path} \
  --adv_lr ${alr} --adv_init_mag ${amag} --adv_max_norm ${anorm} --adv_steps ${asteps} \
  --num_train_epochs 1 \
  --seed 42 \
  --logging_steps 100 --save_steps 100 \
  --probe_mode True \
  --save_path ${save_path}\
  --overwrite_output_dir
}


gpu=0
adv_init_mag=0e-0
for adv_lr in 1e-1
do
for adv_max_norm in 8.1e-1 8.2e-1 8.3e-1 8.4e-1 8.5e-1 8.6e-1 8.7e-1 8.8e-1 8.9e-1 9e-1
  do
    for adv_steps in 15
      do
        echo "adv_lr: ${adv_lr}, adv_init_mag: ${adv_init_mag}, adv_max_norm: ${adv_max_norm}, adv_steps: ${adv_steps}"
        runexp     ${gpu}    ${adv_lr}    ${adv_init_mag}     ${adv_max_norm}     ${adv_steps}
      done
  done
done