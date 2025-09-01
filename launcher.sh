#!/usr/bin/env bash

set -euo pipefail

first_block=0
last_block=36
split='train'

# check HAFH env var set
[ -z "${HAFH}" ] && { echo "HAFH env var not set"; exit 1; }

# Launcher
for i in $(seq -f "%0${#last_block}g" $first_block $last_block); do
    echo "Tokenizing split ${split}, block ${i} of ${last_block}"
    tmux new-window -d -t "focalcodec-${split}" -n "focalcodec_$i" -- bash -c "
        srun --partition a6000 --time=04:00:00 --gres=gpu:1 --qos=gpu-short \
            conda run --live-stream -n focal-codec \
                ${HAFH}/focalcodec/mls.py --split ${split} ${i}; 
                bash" # stay open (tmux window)
    sleep 1
done
