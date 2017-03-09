#!/bin/bash

#$ -N output
#$ -j y

python -m scoop optimize_epileptor_qsub.py

