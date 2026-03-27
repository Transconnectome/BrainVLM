
set +x

cd YOUR_PROJECT_ROOT   #TODO: Change to your own scratch space


module load python
#module load pytorch/1.13.1
module load cpe/23.03

conda activate BLIP_MRI_llava   #TODO: Change to your own conda env

export LIBRARY_PATH=$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=   #TODO: Change to your own scratch space
export HF_HOME=   #TODO: Change to your own scratch space
export TORCH_HOME=   #TODO: Change to your own scratch space


# LLaVA-NeXT-Interleave Multi-turn Comparison Task (Sex or Age)
torchrun --nnodes 1 --nproc_per_node 1 main_BLLaVaNextInterleave_comparison_hf_joint_T1.py

