#! /bin/bash
#$ -S /bin/bash -V
#$ -j y
#$ -cwd
#$ -o /tmp/yassamri2/Tsukuba/hierarchical/code/grid/output
#$ -pe openmp 15
#$ -p -15
#$ -l arch=linux-x64
#$ -l h='Sidious'
#$ -l diskheavy=1
#$ -q yassa.q

cd ../../storage/model_07_C1/
jags untitled_3.script
