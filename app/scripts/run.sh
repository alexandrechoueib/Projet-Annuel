#!/bin/bash
# Description: run performance tests for a compiler
#              and an architecture
# Author: Jean-Michel Richer

if test $# -eq 2 ; then

my_arch=$1
my_comp=$2

else

my_arch=""
while [ "$my_arch" != "32" ] && [ "$my_arch" != "64" ] ; do
	echo "--------------------------------"
	echo "Please enter memory model:"
	echo "- 32 for 32 bits"
	echo "- 64 for 64 bits"
	read -p "memory model (default 32) ? " my_arch
	my_arch=${my_arch:-32}
	echo $my_arch
done

my_comp=""
while [ "$my_comp" != "gnu" ] && [ "$my_comp" != "llvm" ] && [ "$my_comp" != "intel" ] ; do
	echo "--------------------------------"
	echo "Please enter compiler:"
	echo "- gnu for g++ GNU"
	echo "- llvm for clang++"
	echo "- intel for icpc"
	read -p "compiler (default gnu) ? " my_comp
	my_comp=${my_comp:-gnu}
	echo $my_comp
done

fi

if test "$my_comp" = "intel" ; then
    echo "!!!!!!!!!!!"
    echo $arch
    echo $comp
    source /opt/intel/bin/compilervars.sh intel64
fi

mkdir -p tmp

configuration_file="app/config.ezp"
tmp_file=`mktemp`
cat $configuration_file | sed -e "s/MY_ARCHITECTURE=[[:digit:]]\+/MY_ARCHITECTURE=$my_arch/g" | sed -e "s/MY_COMPILER=[[:alpha:]]\+/MY_COMPILER=$my_comp/g" >$tmp_file
cp $tmp_file $configuration_file
rm -rf $tmp_file

make clean --no-print-directory
make config --no-print-directory
make --no-print-directory
if test $? -ne 0 ; then
	exit 1
fi

make performance --no-print-directory
make plots --no-print-directory

