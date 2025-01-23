#!/bin/bash
#SBATCH -J results_01_23
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --cpus-per-task 3
#SBATCH -t 12:00:00
#SBATCH -p genoa
#SBATCH --mail-type=BEGIN,END
#SBATCH -o job_outputs/slurm-%A.out

INPUT_FILE=$HOME/EGG/ancm/data/input_data/visa-4-250.npz
HPARAMS_FILE=$HOME/EGG/ancm/jobs/params_01_23.txt
JOB_FILE=$HOME/EGG/ancm/jobs/job.sh
OUTPUT_DIR=$HOME/runs/results_01_23/

# Loading modules
module load 2024

# Make sure the right Python version is used
echo "$TMPDIR"
python -V
which python
python3 -V
which python3
python -m pip install -e $HOME/EGG
python -m pip install -r $HOME/EGG/ancm/requirements.txt

# Copy the job file and the params into the output dir
mkdir -p $OUTPUT_DIR/job
rsync $HPARAMS_FILE $OUTPUT_DIR/job/
rsync $JOB_FILE $OUTPUT_DIR/job/

# Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
for i in {0..9}; do
    for ((j=48*i; j<48*(i+1); j++)); do
        if ((j <= 475)); then
            srun --ntasks=1 --exclusive \
            python -m ancm.train \
            --data_path $INPUT_FILE \
            --results_folder $OUTPUT_DIR \
            $(head -$j $HPARAMS_FILE | tail -1) &
        fi
    done
    wait
done
