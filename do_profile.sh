#! /usr/bin/bash
nvidia-smi
nvcc -V
nvcc -arch=sm_75 -o temp temp.cu
./temp