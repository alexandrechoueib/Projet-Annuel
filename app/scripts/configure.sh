#!/bin/bash
# Description: configure file for compilation
# Author: Jean-Michel Richer


# -------------------------------------------------------------------
# configure
# -------------------------------------------------------------------

echo "---------------------------"
echo "Configure"
echo "---------------------------"
contains_element () {
  	local e
  	for e in "${@:2}"; do 
		[[ "$e" == "$1" ]] && return 1; 
	done
  	return 0
}

report_possible_error() {
	if test $1 -eq 0 ; then
		shift
		echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
		echo "!!!              Error                !!!"
		echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
		local v=$1
		shift
		echo "found value=$1"
		shift
		echo "possible values are: $*"
		echo "please modify configuration variable '$v'"
		echo "in file app/config.ezp"
		echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
		exit 1
	fi
}

#--------------------------------------------------------------------
# get configuration parameters
#--------------------------------------------------------------------
configuration_file="app/config.ezp"

# determine system

my_opsy=`uname -o | sed -e "s/\//_/g" | tr '[:upper:]' '[:lower:]'`
my_proc=`cat /proc/cpuinfo | grep "^model name" | uniq | cut -d':' -f2 | sed -E -e "s/^[ ]+//g" | sed -E -e "s/[ ]+/ /g" | tr ' ' '-'`


echo "- read configuration parameters from $configuration_file"

my_syst=`cat $configuration_file | grep "^MY_SYSTEM=" | cut -d'=' -f2 | sed -r -e "s/[\ \t]+//g"`
my_vers=`cat $configuration_file | grep "^MY_PROJECT_VERSION=" | cut -d'=' -f2 | sed -r -e "s/[\ \t]+//g"`
my_comp=`cat $configuration_file | grep "^MY_COMPILER=" | cut -d'=' -f2 | sed -r -e "s/[\ \t]+//g"`
my_flav=`cat $configuration_file | grep "^MY_FLAVOR=" | cut -d'=' -f2 | sed -r -e "s/[\ \t]+//g"`
my_arch=`cat $configuration_file | grep "^MY_ARCHITECTURE=" | cut -d'=' -f2 | sed -r -e "s/[\ \t]+//g"`
my_data=`cat $configuration_file | grep "^MY_DATA_SIZE=" | cut -d'=' -f2 | sed -r -e "s/[\ \t]+//g"`
my_maln=`cat $configuration_file | grep "^MY_MEMORY_ALIGNMENT=" | cut -d'=' -f2 | sed -r -e "s/[\ \t]+//g"`
my_rimp=`cat $configuration_file | grep "^MY_REAL_IMPLEMENTATION=" | cut -d'=' -f2 | sed -r -e "s/[\ \t]+//g"`

#--------------------------------------------------------------------
# check if values are correct
#--------------------------------------------------------------------
echo "- check if configuration parameters values are valid"

opsy_allowed=(gnu_linux)
contains_element $my_opsy "${opsy_allowed[@]}"
res=$?
report_possible_error $res "MY_OPERATING_SYSTEM" $my_opsy "${opsy_allowed[@]}"

comp_allowed=(gnu intel llvm)
contains_element $my_comp "${comp_allowed[@]}"
res=$?
report_possible_error $res "MY_COMPILER" $my_comp "${comp_allowed[@]}"

flav_allowed=(release debug advisor)
contains_element $my_flav "${flav_allowed[@]}"
res=$?
report_possible_error $res "MY_FLAVOR" $my_flav "${flav_allowed[@]}"

arch_allowed=(32 64)
contains_element $my_arch "${arch_allowed[@]}"
res=$?
report_possible_error $res "MY_ARCHITECTURE" $my_arch "${arch_allowed[@]}"

data_allowed=(8)
contains_element $my_data "${data_allowed[@]}"
res=$?
report_possible_error $res "MY_DATA_SIZE" $my_data "${data_allowed[@]}"

maln_allowed=(1 16 32)
contains_element $my_maln "${maln_allowed[@]}"
res=$?
report_possible_error $res "MY_MEMORY_ALIGNMENT" $my_maln "${maln_allowed[@]}"

rimp_allowed=(float double)
contains_element $my_rimp "${rimp_allowed[@]}"
res=$?
report_possible_error $res "MY_REAL_IMPLEMENTATION" $my_rimp "${rimp_allowed[@]}"


if test "$my_flav" = "advisor" -a "$my_comp" != "intel" ; then
	echo "if MY_FLAVOR is set to advisor then you should"
	echo "use Intel icpc compiler"
	exit 1
fi


# -------------------------------------
# determine highest vector unit (hvu) on processor
# sse2.0 < sse4.2 < avx2.0
# we use the following notation
#  - sse20 for SSE2 assembly instruction set
#  - sse42 for SSE4.2 assembly instruction set
#  - avx20 for AVX2 assembly instruction set
# -------------------------------------
echo -n "- determine highest vector unit: "


if test "${my_syst}" = "linux" ; then
	cpu_technos=`cat /proc/cpuinfo | grep "flags" | cut -d':' -f2 | head -1`
	has_sse20=`echo $cpu_technos | grep "sse2"`
	has_sse42=`echo $cpu_technos | grep "sse4_2"`
	has_avx20=`echo $cpu_technos | grep "avx2"`
	has_popct=`echo $cpu_technos | grep "popcnt"`
fi

hvu=""
if test -n "${has_avx20}" ; then
	hvu="avx20"
elif test -n "${has_sse42}" ; then
	hvu="sse42"
elif test -n "${has_sse20}" ; then
	hvu="sse20"
fi

echo "$hvu"

asm_cfg_file="src/version_${my_vers}/asm_config.inc"
cpp_cfg_file="src/version_${my_vers}/cpp_config.h"
cmp_cfg_file="app/my_config/compiler.make"

rm -rf $asm_cfg_file $cpp_cfg_file $cmp_cfg_file

# -------------------------------------
# generate assembly configuration file
# -------------------------------------
echo "- write assembly configuration file '$asm_cfg_file'"

if test "$my_opsy" = "gnu_linux" ; then
echo "%define ASM_OS_LINUX" >> $asm_cfg_file
fi

echo "%define ASM_ARCHITECTURE $my_arch" >> $asm_cfg_file
echo "%define ASM_MEMORY_ALIGNMENT $my_maln" >> $asm_cfg_file


if test -n "${has_sse20}" ; then
	echo "%define ASM_SSE20_COMPLIANT 1" >> $asm_cfg_file
fi
if test -n "${has_sse42}" ; then
	echo "%define ASM_SSE42_COMPLIANT 1" >> $asm_cfg_file
fi
if test -n "${has_avx20}" ; then
	echo "%define ASM_AVX20_COMPLIANT 1" >> $asm_cfg_file
fi
if test -n "${has_popct}" ; then
	echo "%define ASM_POPCOUNT_COMPLIANT 1" >> $asm_cfg_file
fi


# -------------------------------------
# generate C++ configuration file
# -------------------------------------
echo "- write C++ configuration file '$cpp_cfg_file'"

echo "// this file was generated by the following script:" >> $cpp_cfg_file
echo "// ./appli/scripts/configure.sh" >> $cpp_cfg_file
echo "#ifndef CPP_CONFIG_H" >> $cpp_cfg_file
echo "#define CPP_CONFIG_H" >> $cpp_cfg_file
if test "$my_opsy" = "gnu_linux" ; then
	echo "#define CPP_OS_LINUX" >> $cpp_cfg_file
fi	
echo "#define CPP_SYSTEM_${my_system_uppercase}" >> $cpp_cfg_file
echo "#define CPP_ARCHITECTURE_"${my_arch}"_BITS" >> $cpp_cfg_file
echo "#define CPP_MEMORY_ALIGNMENT $my_maln" >> $cpp_cfg_file
echo "#define CPP_DATA_SIZE $my_data" >> $cpp_cfg_file


if test -n "${has_sse20}" ; then
	echo "#define CPP_SSE20_COMPLIANT 1" >> $cpp_cfg_file
fi
if test -n "${has_sse42}" ; then
	echo "#define CPP_SSE42_COMPLIANT 1" >> $cpp_cfg_file
fi
if test -n "${has_avx20}" ; then
	echo "#define CPP_AVX20_COMPLIANT 1" >> $cpp_cfg_file
fi

if test "${my_rimp}" = "float" ; then
	echo "typedef float  real;" >> $cpp_cfg_file
else
	echo "typedef double real;" >> $cpp_cfg_file
fi

echo "#endif" >> $cpp_cfg_file


app/scripts/configure_for_${my_comp}.sh $my_syst $my_flav $my_arch $hvu

