#!/bin/bash
# Read-only scheduler checks. Never submits a job or starts inference.
set -euo pipefail
account=aip-sven
partition=gpubase_bygpu_b1
command -v sinfo >/dev/null
command -v sacctmgr >/dev/null
command -v scontrol >/dev/null
echo 'Live partition limits (check MaxTime, CPUs, memory and GPU availability):'
scontrol show partition "$partition"
sinfo -N -p "$partition" -o '%N %G %c %m %a %l'
echo 'Account associations (verify aip-sven and permitted partition/QOS):'
sacctmgr -nP show assoc where user="$USER" account="$account" format=Cluster,Account,Partition,QOS
echo 'Resolve inherited limits / allocation policy with your administrator if these outputs are ambiguous.'
echo 'Before submission confirm 1 GPU, 4 CPUs, 32G and 30 minutes are allowed.'
echo 'Then use sbatch --test-only with the exact submission command to validate scheduler acceptance.'
