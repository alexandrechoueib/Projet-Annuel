#!/bin/bash

if test $# -eq 0 ; then
	exit 0
fi

my_syst=$1
my_flav=$2
my_arch=$3
hvu=$4

if test "$hvu" = "avx20" ; then
	oflags="-mavx2"
elif test "$hvu" = "sse42" ; then
	oflags="-msse4.2"
elif test "$hvu" = "sse20" ; then
	oflags="-msse2"
fi

cpu_technos=`cat /proc/cpuinfo | grep "flags" | cut -d':' -f2 | head -1`
has_popct=`echo $cpu_technos | grep "popcnt"`
	
if test -n "${has_popct}" ; then
	oflags="${oflags} -mpopcnt"
fi

if test "$my_flav" = "release" ; then
	oflags="-O3 -ftree-vectorize $oflags -fopenmp"
else
	oflags="-ggdb $oflags"
fi

if test "$my_syst" = "linux" ; then
	if test "$my_arch" = "32" ; then
		cflags="$cflags -m32"
	else
		cflags="$cflags -m64"
	fi
fi


output="app/compiler/my_compiler.mk"
rm -rf $output

echo "- write $output for extra flags"
echo "CFLAGS += $cflags" > $output
echo "OFLAGS += $oflags" >> $output

