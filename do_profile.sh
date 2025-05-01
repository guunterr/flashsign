#! /usr/bin/bash
nvidia-smi
mkdir out
rm out/*
datetime=$2
nvcc -arch=sm_80 -lcublas -g -DNDEBUG --generate-line-info -O3 --ptxas-options=-v -o out/temp src/temp.cu | tee out/compiler_log_$datetime.txt
./out/temp t $1 1024 1 0.001 > out/test_$datetime.txt
./out/temp r $1 10 20 | tee out/benchmark_$datetime.txt
ncu -f --launch-skip 1 --set full --export out/profile_$datetime ./out/temp r $1 1 1
 