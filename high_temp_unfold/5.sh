#!/bin/bash
#BSUB -J miniprotein
#BSUB -n 1
#BSUB -q gpuqueue
#BSUB -gpu "num=1"
#BSUB -R span[ptile=1]
#BSUB -R rusage[mem=10]
#BSUB -W 48:00
#BSUB -o /home/rafal.wiewiora/job_outputs/%J.stdout
#BSUB -eo /home/rafal.wiewiora/job_outputs/%J.stderr
 
cd $LS_SUBCWD
python prepare-for-fah.py 5
