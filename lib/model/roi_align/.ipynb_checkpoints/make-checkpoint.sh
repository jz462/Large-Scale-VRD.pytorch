#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
# # By Ji on 08/20/2018
# nvcc -c -o roi_align_rel_kernel.cu.o roi_align_rel_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
python3 build.py
