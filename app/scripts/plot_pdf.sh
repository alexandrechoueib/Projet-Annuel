#!/bin/bash
# Description: print results as graphics in pdf format
# Author: Jean-Michel Richer

if test $# -ne 1 ; then
	echo "error: one parameter required ARCH "
	echo "- ARCH = 32 or 64"
	exit 1
fi


my_arch=$1
suffix="${my_arch}"
main=`ls build/bin/*.exe | head -1`

configuration_file="app/config.ezp"

my_syst=`cat $configuration_file | grep "MY_SYSTEM" | cut -d'=' -f2 | tr -d ' '`
my_comp=`cat $configuration_file | grep "MY_COMPILER" | cut -d'=' -f2 | tr -d ' '`

cpu_name=`app/scripts/hardware_cpu_name.sh`
cpu_name_gp=`echo $cpu_name | tr '_' ' '`

date=`date +%Y_%m_%d`
results_file="results/data/${cpu_name}_${my_syst}_${my_comp}_${my_arch}.txt"

mkdir -p results/plots/${cpu_name}/${my_arch}/${my_comp}

# get test parameters
source ./app/performance_parameters.sh

mkdir -p tmp

for treatment in $treatments ; do

	iterations=`echo $iterations_per_treatment | tr '|' '\n' | grep $treatment | cut -d':' -f2`
	lengths=`echo $lengths_per_treatment | tr '|' '\n' | grep $treatment | cut -d':' -f2`
	
	#echo "iterations=$iterations"
	#echo "lengths tested=$lengths"
	
	results_file="results/data/${cpu_name}_${my_syst}_${my_comp}_${my_arch}_${treatment}.txt"
	
	output="results/plots/${cpu_name}/${my_arch}/${my_comp}/plot.gps"
	
	echo "- png for $treatment"

	treatment_str=`echo $treatment | sed -e "s/\_/ /g"`
	echo "set datafile separator \";\"" > $output
	echo "set title \"${cpu_name_gp} - ${treatment_str} - ${my_arch} bits\"" >> $output
	echo "set output \"results/plots/${cpu_name}/${my_arch}/${my_comp}/${cpu_name}_${my_syst}_${my_comp}_${my_arch}_${treatment}.pdf\"" >>$output
	echo "set terminal pdfcairo color font \"Verdana,9\"" >>$output
	
	echo "load 'app/gnuplot/my.pal'" >>$output
	echo "set grid" >>$output
	
	echo -n "plot " >tmp/tmp_gp.txt
	n=`echo $implementations | wc -w`
	p=`expr $n - 1`
	
	i=1
	while [ $i -le $n ] ; do
	    implementation=`echo $implementations | cut -d' ' -f$i`
	    cat $results_file | grep "^$implementation;" >tmp/tmp_$implementation.txt
	    echo -n "\"tmp/tmp_$implementation.txt\" using 3:5 with lines ls $i title \"$implementation\"" >> tmp/tmp_gp.txt
	    i=`expr $i + 1`
	    if test $i -le $n ; then
	       echo -n ", " >>tmp/tmp_gp.txt
	    fi
	done

	cat tmp/tmp_gp.txt >> $output
	
	gnuplot $output
	
done	
