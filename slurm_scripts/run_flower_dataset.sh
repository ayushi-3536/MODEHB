#!/bin/bash -l
#SBATCH -p mlhiwidlc_gpu-rtx2080 #partition
#SBATCH --mem 12000 # memory pool for all cores (4GB)
#SBATCH -t 16:24:00 # time (D-HH:MM)
#SBATCH -c 1 # number of cores
#SBATCH -a 1-2 # array size
#SBATCH --gres=gpu:1  # reserves one GPU
#SBATCH -D /work/dlclarge1/sharmaa-modehb # Change working_dir
#SBATCH -o /work/dlclarge1/sharmaa-modehb/logs/flower_exp.%A.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e /work/dlclarge1/sharmaa-modehb/logs/flower_exp.%A.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sharmaa@informatik.uni-freiburg.de
# Activate virtual env so that run_experiment can load the correct packages

cd $(ws_find modehb)
cd MODEHB
source activate virtualEnvironment

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"

if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
           mkdir -p ./example/job-$SLURM_ARRAY_TASK_ID &&  python3 -m src.flower_dataset --runtime 10 --run_id 1 --output_path '/work/dlclarge1/sharmaa-modehb/logs' --seed 2
              exit $?
fi