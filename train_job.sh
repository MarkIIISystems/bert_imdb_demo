#!/bin/bash
#SBATCH --output /home/michaela/bert_example/sbatch/outs/%j.out
#SBATCH --error /home/michaela/bert_example/sbatch/outs/%j.err
#SBATCH -p dgxq
#SBATCH --gres gpu:7
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=7

module add cuda11.8

source /home/michaela/leash/leash-venv/bin/activate

NODES=$SLURM_NNODES
GPUS=$SLURM_GPUS_ON_NODE
BATCH_SIZE=128
NUM_WORKERS=0
EPOCHS=1
JOB_NUM=$SLURM_JOBID

srun python ~/bert_example/scripts/bert-train.py $NODES $GPUS $BATCH_SIZE $NUM_WORKERS $JOB_NUM $EPOCHS
