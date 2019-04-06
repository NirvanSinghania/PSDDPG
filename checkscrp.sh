#!/bin/bash

n=$1
name="^network"$n"DDPG.ckpt.*meta$"
output=$(ls weights | grep $name -m 1 | cut -d "." -f 1-2)
echo "model_checkpoint_path: "\"$output\" >  weights/checkpoint
echo "all_model_checkpoint_paths: "\"$output\" >>  weights/checkpoint

