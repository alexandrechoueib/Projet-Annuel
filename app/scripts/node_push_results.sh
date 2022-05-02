#!/bin/bash
# Description: send results of performance tests to the server 
# Author: Jean-Michel Richer

echo "========================================="
echo "Push results to the server"
echo "========================================="

configuration_file="app/config.ezp"

my_user=`cat $configuration_file | grep "MY_USER=" | cut -d'=' -f2`
my_server=`cat $configuration_file | grep "MY_SERVER=" | cut -d'=' -f2`
my_server_dir=`cat $configuration_file | grep "MY_SERVER_DIRECTORY=" | cut -d'=' -f2`

cpu_name=`app/scripts/hardware_cpu_name.sh`


tar -cvzf results/${cpu_name}.tgz results/data/${cpu_name}* results/hardware/${cpu_name} results/plots/${cpu_name} 
uri="${my_server}:${my_server_dir}"


# we will use rsync over ssh
echo "files will be copied to $server_name"

rsync -arvzhe ssh results/${cpu_name}.tgz $uri
