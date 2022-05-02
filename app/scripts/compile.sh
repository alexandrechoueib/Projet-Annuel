#!/bin/sh

PROJECT_DIR=$1
shift
BUILD_DIR=$1
shift
LIB_SRC_DIR=$1
shift
MODULES=$*


# copy include files

for module in $MODULES ; do 
	echo "- copy include files for MODULE $module" 
	mkdir -p $BUILD_DIR/include/$module
	cp $LIB_SRC_DIR/$module/*.h  $BUILD_DIR/include/$module 
done


for module in $MODULES ; do 
	echo "============================================================" 
	echo "COMPILE MODULE $module" 
	echo "============================================================" 
	cd $LIB_SRC_DIR/$module
	make --no-print-directory PROJECT_DIR=$PROJECT_DIR BUILD_DIR=$BUILD_DIR MODULE=$module
	cd $PROJECT_DIR 
done
