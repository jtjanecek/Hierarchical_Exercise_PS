#! /bin/bash
#$ -S /bin/bash -V
#$ -j y
#$ -cwd
#$ -o /tmp/yassamri2/Tsukuba/hierarchical/code/grid/output
#$ -pe openmp 15
#$ -p -15
#$ -l arch=linux-x64
#$ -l diskheavy=1
#$ -q shared.q,yassa.q

/tmp/yassamri/Software/John_Anaconda/bin/python /tmp/yassamri2/Tsukuba/hierarchical/CodaFormatter/Chain.py --niter 100000 --nthin 1 --chain /tmp/yassamri2/Tsukuba/hierarchical/storage/model_07_C1/samples_$1_chain1.txt --index /tmp/yassamri2/Tsukuba/hierarchical/storage/model_07_C1/samples_$1_index.txt --out /tmp/yassamri2/Tsukuba/hierarchical/storage/model_07_C1/
