#!/bin/bash

if test $# -ne 3 ; then
	echo "error: three parameters required CPU METHOD SIZE"
	echo "please use one of the followings for CPU: "
	ls results/hardware/
	nb_methods=`build/bin/main.exe -c | grep "functions.count" | cut -d'=' -f2`
	echo "- METHOD = 1 to $nb_methods"
	echo "- SIZE is one of 1024 2048 4096 8192 16384"
	exit 1
fi

cpu=$1
method=$2
size=$3

opsy="linux"
compilers="gnu intel llvm"
architectures="32 64"

for c in $compilers ; do
	for a in $architectures ; do
		input_file="results/data/${cpu}_${opsy}_${c}_${a}.txt"
		if test -f $input_file ; then
			line=`cat $input_file | grep "^$method ; $size ; "`
			echo "$c $a $line"
		fi 
	done
done

