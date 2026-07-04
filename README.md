# BrainSketcher

# Stage I 
## Training on One GPU 
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port 15666 \
    code/stageA1_eeg_pretrain_copy.py

## Traning on Multiple GPUs
python -m torch.distributed.launch --nproc_per_node=2 \
    code/stageA1_eeg_pretrain_copy.py

# Stage II 
