#!/bin/bash
# Description: run performance tests for all compilers
#              and architectures
# Author: Jean-Michel Richer

app/scripts/hardware_cpu_info_auto.sh
app/scripts/hardware_mem_info_auto.sh

gnu_compiler=`which g++`
if test -n "$gnu_compiler" ; then  
    ./app/scripts/run.sh 32 gnu
    ./app/scripts/run.sh 64 gnu
fi

llvm_compiler=`which clang++`
if test -n "$llvm_compiler" ; then
    ./app/scripts/run.sh 32 llvm
    ./app/scripts/run.sh 64 llvm
fi

if test -f /opt/intel/bin/compilervars.sh ; then
    ./app/scripts/run.sh 64 intel
fi

app/scripts/node_push_results.sh
