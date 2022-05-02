CFLAGS=-Wall -std=c++14 
OFLAGS=-O3 -ftree-vectorize -march=native -fopenmp 

AR=ar
RANLIB=ranlib
CC=clang
CPP=clang++
NASM=nasm -f elf64
