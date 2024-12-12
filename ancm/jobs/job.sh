#!/bin/bash
#SBATCH -J experiments_1
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -p genoa
#SBATCH --exclusive
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=cezary.klamra@student.uva.nl
#SBATCH --array=1-760%96

JOB_FILE=$HOME/EGG/ancm/jobs/params.txt
PARAMS_FILE=$HOME/EGG/ancm/jobs/job.sh
OUTPUT_DIR=$HOME/runs/experiments_1

# Loading modules
module load 2022
module load Python/3.9.5-GCCcore-10.3.0

# Make sure the right Python version is used
echo "$TMPDIR"
python -V
which python
python3 -V
which python3

# Copy the job file and the params into the output dir
if [ -d my_dir ]; then
  mkdir -p $OUTPUT_DIR/job
else
  exit 1
fi
rsync $HPARAMS_FILE $OUTPUT_DIR/job
rsync $JOB_FILE $OUTPUT_DIR/job

# Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
srun python -m ancm.train \           
  --dump_results_folder $OUTPUT_DIR
  $(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1)
