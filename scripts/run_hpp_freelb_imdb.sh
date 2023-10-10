#!/usr/bin/env bash

function runexp {
gpu=${1}      # The GPU you want to use
alr=${2}      # Step size of gradient ascent
amag=${3}     # Magnitude of initial (adversarial?) perturbation
anorm=${4}    # Maximum norm of adversarial perturbation
asteps=${5}   # Number of gradient ascent steps for the adversary

root_path=tensors/
file_name=freelb_freelb_imdb_alr${alr}_amag${amag}_anm${anorm}_as${asteps}
save_path=${root_path}${file_name}.pt
output_path=hpp_outputs/${file_name}

CUDA_VISIBLE_DEVICES=${gpu} python ../model/freelb.py \
  --model_name_or_path {{YOUR_MODEL_PATH}} \
  --dataset_name imdb \
  --do_train \
  --max_seq_length 512 \
  --pad_to_max_length True \
  --per_device_train_batch_size 32 \
  --output_dir ${output_path} \
  --adv_lr ${alr} --adv_init_mag ${amag} --adv_max_norm ${anorm} --adv_steps ${asteps} \
  --num_train_epochs 1 \
  --seed 42 \
  --logging_steps 100 --save_steps 100 \
  --probe_mode True \
  --save_path ${save_path}\
  --overwrite_output_dir
}


gpu=1
adv_init_mag=0e-0
for adv_lr in 1e-1
do
  for adv_max_norm in 1e-1 1.1e-1 1.2e-1 1.3e-1 1.4e-1 1.5e-1 1.6e-1 1.7e-1 1.8e-1 1.9e-1 2e-1 2.1e-1 2.2e-1 2.3e-1 2.4e-1 2.5e-1 2.6e-1 2.7e-1 2.8e-1 2.9e-1 3e-1 3.1e-1 3.2e-1 3.3e-1 3.4e-1 3.5e-1 3.6e-1 3.7e-1 3.8e-1 3.9e-1 4e-1 4.1e-1 4.2e-1 4.3e-1 4.4e-1 4.5e-1 4.6e-1 4.7e-1 4.8e-1 4.9e-1 5e-1 5.1e-1 5.2e-1 5.3e-1 5.4e-1 5.5e-1 5.6e-1 5.7e-1 5.8e-1 5.9e-1 6e-1 6.1e-1 6.2e-1 6.3e-1 6.4e-1 6.5e-1 6.6e-1 6.7e-1 6.8e-1 6.9e-1 7e-1 7.1e-1 7.2e-1 7.3e-1 7.4e-1 7.5e-1 7.6e-1 7.7e-1 7.8e-1 7.9e-1 8e-1
  do
    for adv_steps in 15
      do
        echo "adv_lr: ${adv_lr}, adv_init_mag: ${adv_init_mag}, adv_max_norm: ${adv_max_norm}, adv_steps: ${adv_steps}"
        runexp     ${gpu}    ${adv_lr}    ${adv_init_mag}     ${adv_max_norm}     ${adv_steps}
      done
  done
done