#!/bin/bash
#SBATCH -J maxlen_2
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -p genoa
#SBATCH --requeue
#SBATCH --exclusive
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=cezary.klamra@student.uva.nl

# Loading modules
module load 2022
module load Python/3.9.5-GCCcore-10.3.0
 
# Create output directory on scratch
mkdir "$TMPDIR"/runs
 
python -m pip install -r $HOME/EGG/ancm/requirements39.txt
python -m pip install $HOME/EGG

# Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
python -m ancm.run_all --output_dir "$TMPDIR"/runs --batch_size 96
 
# Copy output directory from scratch to home
cp -r "$TMPDIR"/runs $HOME
