#!/bin/bash -l

#$ -N output
#$ -j y
#$ -v target_file=/usr3/graduate/eds2/eeg_targets/target.pkl
#$ -v eeg_file=/usr3/graduate/eds2/eeg_targets/outputEdf_0.edf
#$ -t 1-10

python optimize_epileptor_qsub.py $SGE_TASK_ID $target_file;

