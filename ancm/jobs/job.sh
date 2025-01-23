#!/bin/bash
#SBATCH -J results_01_23
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 12:00:00
#SBATCH -p genoa
#SBATCH --mail-type=BEGIN,END
#SBATCH -o job_outputs/slurm-%A.out

INPUT_FILE=$HOME/EGG/ancm/data/input_data/visa-4-250.npz
HPARAMS_FILE=$HOME/EGG/ancm/jobs/params_rf.txt
JOB_FILE=$HOME/EGG/ancm/jobs/job.sh
OUTPUT_DIR=$HOME/runs/results_01_14/

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
for idx in {1..475}
do
  srun --ntasks=1 --exclusive --mem=6G \
    python -m ancm.train \
    --data_path $INPUT_FILE \
    --results_folder $OUTPUT_DIR \
    $(head -$idx $HPARAMS_FILE | tail -1) &
done
wait
