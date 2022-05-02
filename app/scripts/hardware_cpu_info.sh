#!/bin/bash
# Description: get information about cpu and save it
# Author: Jean-Michel Richer

echo "- retrieve cpu information"
# get cpu identifier as name
cpu_name=`app/scripts/hardware_cpu_name.sh`

output_dir="results/hardware/${cpu_name}"
output_file="${output_dir}/description"

if test -f $output_file ; then
    echo "-! warning ! hardware information already gathered"
    exit 0
fi

mkdir -p $output_dir

echo "-! warning ! connect as root in order to obtain"
echo "             information from dmidecode"

echo "found CPU: $cpu_name"

sudo dmidecode -t 4,7,16,17 >$output_file
