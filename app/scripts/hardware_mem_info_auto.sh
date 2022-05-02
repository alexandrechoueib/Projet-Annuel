#!/bin/sh
# Description: get information about memory and save it
#              note: we do not use dmidecode or lshw 
#              as it requires to have root priviledges to get
#              full information
# Author: Jean-Michel Richer

cpu_name=`app/scripts/hardware_cpu_name.sh`

free -h > out.txt

output_file="results/hardware/${cpu_name}/memory.txt"

mkdir -p `dirname $output_file`

echo "====================================" > $output_file
echo "${cpu_name}" >> $output_file
echo "====================================" >> $output_file
cat out.txt >> $output_file
cat out.txt | grep DDR >> $output_file

rm out.txt
echo "Output file is $output_file"
cat $output_file

