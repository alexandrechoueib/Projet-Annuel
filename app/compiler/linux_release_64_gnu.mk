CFLAGS=-Wall -std=c++14 
OFLAGS=-O3 -ftree-vectorize -march=native -fopenmp 
CUFLAGS= -gencode arch=compute_61,code=sm_61 \
	-gencode arch=compute_50,code=sm_50 \
	-gencode arch=compute_35,code=sm_35 \
	--compiler-options "-Wno-vla -O2 -I./src" \
	-std=c++11

AR=ar
RANLIB=ranlib
CC=gcc
CPP=g++
NASM=nasm -f elf64
NVCC=nvcc
