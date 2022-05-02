# you can add the following to OFLAGS for vectorization
#    -xCORE-AVX2 for AVX 2.0
# or -xSSE4.2    for SSE 4.2
# or -xSSE2      for SSE 2.0

CFLAGS=-Wall -std=c++14
OFLAGS= -fopenmp 

AR=ar
RANLIB=ranlib
CC=icc
CPP=icpc
NASM=nasm -f elf64
