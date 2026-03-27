
set +x

cd /pscratch/sd/h/heehaw/BLIP_MRI/project   #TODO: Change to your own scratch space


module load python
#module load pytorch/1.13.1
module load cpe/23.03

conda activate /pscratch/sd/h/heehaw/anaconda/BLIP_MRI   #TODO: Change to your own conda env
# conda activate py39
# pip install timm
#export MASTER_ADDR=`/bin/hostname -s`
#export MASTER_PORT=29500
#export MASTER_PORT=$(shuf -i 29500-65535 -n 1)

export LIBRARY_PATH=$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/pscratch/sd/h/heehaw   #TODO: Change to your own scratch space
export HF_HOME=/pscratch/sd/h/heehaw/huggingface   #TODO: Change to your own scratch space
export TORCH_HOME=/pscratch/sd/h/heehaw/   #TODO: Change to your own scratch space


# #recent version (24.3.30)
torchrun --nnodes 1 --nproc_per_node 1 main_Bblip_t5_hf_joint.py


# --limit_training_samples 1000
# --mask_patch_size 12 12 12 20
 
