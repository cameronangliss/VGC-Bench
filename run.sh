#!/bin/bash
if [[ $PATH != "/scratch/cluster/cangliss/repos/UT-masters-thesis/bin:"* ]]; then
    export PATH="/scratch/cluster/cangliss/repos/UT-masters-thesis/bin:$PATH"
fi

python dexter/pretrain.py
while true; do python dexter/train.py > debug.log 2>&1; sleep 10; done
