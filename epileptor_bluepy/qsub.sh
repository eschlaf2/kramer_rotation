#!/bin/bash

#$ -N output
#$ -j y

for i in {1..10};
do
    python -m scoop optimize_epileptor_qsub.py $i;
done

