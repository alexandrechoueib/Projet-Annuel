#!/bin/bash

if test $# -eq 0 ; then
	exit 0
fi

my_syst=$1
my_flav=$2
my_arch=$3
hvu=$4


if test "$hvu" = "avx20" ; then
	oflags="-xCORE-AVX2"
elif test "$hvu" = "sse42" ; then
	oflags="-xSSE4.2"
elif test "$hvu" = "sse20" ; then
	oflags="-xSSE2"
fi

cpu_technos=`cat /proc/cpuinfo | grep "flags" | cut -d':' -f2 | head -1`


if test "$my_flav" = "release" ; then
	oflags="${oflags} -funroll-loops  -ipo -diag-error-limit=5 -diag-enable=all "
elif test "$my_flav" = "advisor" ; then
	oflags="${oflags} -g -qopt-report=5 -simd -restrict -qopt-report=1 -qopt-report-phase=vec -qopt-report-file=stdout -diag-error-limit=5"
else	
	oflags="${oflags} -w5"
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
