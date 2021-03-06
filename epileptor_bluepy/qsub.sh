#!/bin/bash

#$ -N output
#$ -j y
#$ -v target_file=/usr3/graduate/eds2/eeg_targets/target.pkl
#$ -v eeg_file=/usr3/graduate/eds2/eeg_targets/outputEdf_0.edf

# for i in {1..10};
for i in 1;
do
    python -m scoop optimize_epileptor_qsub.py $i $eeg_file;
done

