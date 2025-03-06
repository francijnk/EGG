#!/bin/bash
#SBATCH -J o5_gpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH -t 4:30:00
#SBATCH -p gpu
#SBATCH --mail-type=BEGIN,END
#SBATCH -o job_outputs/slurm-%A.out

# Loading modules
module load 2024

cd $HOME/EGG/
INPUT_FILE=$HOME/EGG/ancm/data/input_data/obverter-5-160-64.npz
HPARAMS_FILE=$HOME/EGG/ancm/jobs/params_img5.txt
JOB_FILE=$HOME/EGG/ancm/jobs/job_obverter.sh
OUTPUT_DIR=$HOME/EGG/ancm/runs/03_04/obverter_1/
CUBLAS_WORKSPACE_CONFIG=:4096:2

cp $INPUT_FILE $TMPDIR/input_data

# Make sure the right Python version is used
echo "$TMPDIR"
python3 -V
which python3
# python -m pip install -e $HOME/EGG
# python -m pip install -r $HOME/EGG/ancm/requirements.txt
python3 -uc "import torch; print(torch.cuda.mem_get_info())"
# python3 -m ancm.data.obverter -d 4 --n_samples_train 128 --n_samples_test 20 --n_img 100 --resolution 64

# Copy the job file and the params into the output dir
mkdir -p $OUTPUT_DIR/job
rsync $HPARAMS_FILE $OUTPUT_DIR/job/
rsync $JOB_FILE $OUTPUT_DIR/job/

# Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
for i in {0..24}; do
    for ((j=16*i+1; j<=16*(i+1); j++)); do
        if ((j <= 190)); then
            CUDA_VISIBLE_DEVICES=$(($j % 4)) \
            python -m ancm.train --wandb_group 03_04_img \
            --data_path $TMPDIR/input_data \
            --results_folder $OUTPUT_DIR \
            --image_input \
            $(head -$j $HPARAMS_FILE | tail -1) &
            # srun --exclusive \
            # python -m ancm.train --wandb_group 03_04_img  \
            # --data_path $INPUT_FILE \
            # --results_folder $OUTPUT_DIR \
            # $(head -$j $HPARAMS_FILE | tail -1) &
        fi
    done
    wait
done
