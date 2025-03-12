#! /usr/bin/bash
ls
nvcc -o temp temp.cu
./temp
ls