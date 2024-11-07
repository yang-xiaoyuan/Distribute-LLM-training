#!/bin/bash

#SBATCH --nodes=1                   # 节点数
#SBATCH --ntasks=4                  # 任务数
#SBATCH --partition=gpu             # 分区名称（根据集群修改）
#SBATCH --gres=gpu:4                # 设置使用的GPU数

module load nvidia/cuda/12.2
module load mpich/3.4.1-gcc9.3      # 加载gcc-5版本以上

deepspeed   --num_nodes=1          \
            --num_gpus=4           \
            --launcher slurm       \
            run_bert.py 


