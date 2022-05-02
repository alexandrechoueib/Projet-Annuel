#!/bin/bash

# Description: return cpu identifier as name gathering information
#              from /proc/cpuinfo.model_name
# Author: Jean-Michel Richer

cpu_name=`cat /proc/cpuinfo | grep "^model name" | head -1 | cut -d':' -f2 | sed -e "s/([RTMrtm]\+)//g" |  sed -e "s/^[ ]\+//" | sed -e "s/[ ]\+/ /g" | sed -e "s/[ ]\+CPU[ @]\+/ /g" | tr ' ' '_'`

echo $cpu_name

