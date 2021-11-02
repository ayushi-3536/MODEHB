#!/bin/bash -l
#SBATCH -p mlhiwidlc_gpu-rtx2080 #partition
#SBATCH --mem 4000 # memory pool for all cores (4GB)
#SBATCH -t 0-4:00:00 # time (D-HH:MM)
#SBATCH -c 1 # number of cores
#SBATCH -a 1-10 # array size
#SBATCH --gres=gpu:1  # reserves one GPU
#SBATCH -D /work/dlclarge1/sharmaa-modehb # Change working_dir
#SBATCH -o /work/dlclarge1/sharmaa-modehb/fashion_logs/flower_exp.%A.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e /work/dlclarge1/sharmaa-modehb/fashion_logs/flower_exp.%A.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sharmaa@informatik.uni-freiburg.de
# Activate virtual env so that run_experiment can load the correct packages

cd $(ws_find modehb)
cd MODEHB
pip install -r requirements.txt

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"

if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
           mkdir -p ./example/job-$SLURM_ARRAY_TASK_ID &&  python3 -m src.examples.fashion_dataset --runtime 14400 --run_id 3 --output_path '/work/dlclarge1/sharmaa-modehb/fashion_logs' --seed 3
              exit $?
fi
if [ 2 -eq $SLURM_ARRAY_TASK_ID ]; then
           mkdir -p ./example/job-$SLURM_ARRAY_TASK_ID &&  python3 -m src.examples.fashion_dataset --runtime 14400 --run_id 4 --output_path '/work/dlclarge1/sharmaa-modehb/fashion_logs' --seed 4
              exit $?
fi
if [ 3 -eq $SLURM_ARRAY_TASK_ID ]; then
           mkdir -p ./example/job-$SLURM_ARRAY_TASK_ID &&  python3 -m src.examples.fashion_dataset --runtime 14400 --run_id 5 --output_path '/work/dlclarge1/sharmaa-modehb/fashion_logs' --seed 5
              exit $?
fi
if [ 4 -eq $SLURM_ARRAY_TASK_ID ]; then
           mkdir -p ./example/job-$SLURM_ARRAY_TASK_ID &&  python3 -m src.examples.fashion_dataset --runtime 14400 --run_id 6 --output_path '/work/dlclarge1/sharmaa-modehb/fashion_logs' --seed 6
              exit $?
fi
if [ 5 -eq $SLURM_ARRAY_TASK_ID ]; then
           mkdir -p ./example/job-$SLURM_ARRAY_TASK_ID &&  python3 -m src.examples.fashion_dataset --runtime 14400 --run_id 7 --output_path '/work/dlclarge1/sharmaa-modehb/fashion_logs' --seed 7
              exit $?
fi
if [ 6 -eq $SLURM_ARRAY_TASK_ID ]; then
           mkdir -p ./example/job-$SLURM_ARRAY_TASK_ID &&  python3 -m src.examples.fashion_dataset --runtime 14400 --run_id 8 --output_path '/work/dlclarge1/sharmaa-modehb/fashion_logs' --seed 8
              exit $?
fi
if [ 7 -eq $SLURM_ARRAY_TASK_ID ]; then
           mkdir -p ./example/job-$SLURM_ARRAY_TASK_ID &&  python3 -m src.examples.fashion_dataset --runtime 14400 --run_id 9 --output_path '/work/dlclarge1/sharmaa-modehb/fashion_logs' --seed 9
              exit $?
fi
if [ 8 -eq $SLURM_ARRAY_TASK_ID ]; then
           mkdir -p ./example/job-$SLURM_ARRAY_TASK_ID &&  python3 -m src.examples.fashion_dataset --runtime 14400 --run_id 10 --output_path '/work/dlclarge1/sharmaa-modehb/fashion_logs' --seed 10
              exit $?
fi
if [ 9 -eq $SLURM_ARRAY_TASK_ID ]; then
           mkdir -p ./example/job-$SLURM_ARRAY_TASK_ID &&  python3 -m src.examples.fashion_dataset --runtime 14400 --run_id 11 --output_path '/work/dlclarge1/sharmaa-modehb/fashion_logs' --seed 11
              exit $?
fi
if [ 10 -eq $SLURM_ARRAY_TASK_ID ]; then
           mkdir -p ./example/job-$SLURM_ARRAY_TASK_ID &&  python3 -m src.examples.fashion_dataset --runtime 14400 --run_id 12 --output_path '/work/dlclarge1/sharmaa-modehb/fashion_logs' --seed 12
              exit $?
fi