#!/bin/bash
#SBATCH -J visa17_234
#SBATCH -N 2
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --cpus-per-task 6
#SBATCH -t 3:00:00
#SBATCH -p genoa
#SBATCH --mail-type=BEGIN,END
#SBATCH -o job_outputs/slurm-%A.out

# Loading modules
module load 2024

cd $HOME/EGG/
INPUT_FILE=$HOME/EGG/ancm/data/input_data/visa-4-256.npz
HPARAMS_FILE=$HOME/EGG/ancm/jobs/final/visa17_234.txt
JOB_FILE=$HOME/EGG/ancm/jobs/job_visa.sh
OUTPUT_DIR=$HOME/runs/results_02_05_visa/
CUBLAS_WORKSPACE_CONFIG=:4096:2

cp $INPUT_FILE $TMPDIR/visa.npz
ls $TMPDIR

# Make sure the right Python version is used
echo "$TMPDIR"
python3 -V
cd $HOME/EGG
which python3
python -m pip install -e $HOME/EGG
python -m pip install -r $HOME/EGG/ancm/requirements.txt
# python3 -m ancm.data.visa -d 4 --n_samples_train 256

# Copy the job file and the params into the output dir
mkdir -p $OUTPUT_DIR/job
rsync $HPARAMS_FILE $OUTPUT_DIR/job/
rsync $JOB_FILE $OUTPUT_DIR/job/

# Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
for i in {0..10}; do
    for ((j=96*i+1; j<=96*(i+1); j++)); do
        if ((j <= 663)); then
            srun --ntasks=1 --exclusive \
            python -m ancm.train --wandb_group 02_05_visa  \
            --n_permutations_train 3 \
            --data_path $TMPDIR/visa.npz \
            --results_folder $OUTPUT_DIR \
            $(head -$j $HPARAMS_FILE | tail -1) &
        fi
    done
    wait
done
