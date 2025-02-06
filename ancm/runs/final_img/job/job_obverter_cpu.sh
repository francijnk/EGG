#!/bin/bash
#SBATCH -J o5_5
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --cpus-per-task 12
#SBATCH -t 12:00:00
#SBATCH -p genoa
#SBATCH --mail-type=BEGIN,END
#SBATCH -o job_outputs/slurm-%A.out

# Loading modules
module load 2024

cd $HOME/EGG/
pwd
INPUT_FILE=$HOME/EGG/ancm/data/input_data/obverter-5-100-64.npz
HPARAMS_FILE=$HOME/EGG/ancm/jobs/finalparams/obv5.txt
JOB_FILE=$HOME/EGG/ancm/jobs/job_obverter_cpu.sh
OUTPUT_DIR=$HOME/runs/final/obv5/
CUBLAS_WORKSPACE_CONFIG=:4096:2

cp $INPUT_FILE $TMPDIR/obverter.npz
ls $TMPDIR

# Make sure the right Python version is used
echo "$TMPDIR"
python3 -V
cd $HOME/EGG
which python3
# python -m pip install -e $HOME/EGG
# python -m pip install -r $HOME/EGG/ancm/requirements.txt
# python3 -m ancm.data.visa -d 4 --n_samples_train 256

# Copy the job file and the params into the output dir
mkdir -p $OUTPUT_DIR/job
rsync $HPARAMS_FILE $OUTPUT_DIR/job/
rsync $JOB_FILE $OUTPUT_DIR/job/

# Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
for i in {0..10}; do
    for ((j=32*i+1; j<=32*(i+1); j++)); do
        if ((j <= 65)); then
            srun --ntasks=1 --exclusive \
            python -m ancm.train --wandb_group 02_05_obverter  \
            --temperature 2 \
            --data_path $TMPDIR/obverter.npz \
            --results_folder $OUTPUT_DIR \
            --image_input \
            $(head -$j $HPARAMS_FILE | tail -1) &
        fi
    done
    wait
done
