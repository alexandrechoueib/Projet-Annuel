#!/bin/bash

n=1
while [ $n -lt 20000 ] ; do
	echo "coucou $n"
	n=`expr $n + 1`
done

file_name=`date "+%Y%m%d-%H%M"`

echo "coucou $file_name" >$file_name

rsync -arvzhe ssh $file_name richer@192.168.0.1:/home/richer/tmp
 


