#!/bin/sh
if test $# -le 5 ; then
	echo "run_tests/sh needs at least 3 arguments"
	exit 1
fi


PROJECT_DIR=$1
shift
BUILD_DIR=$1
shift
TESTS_DIR=$1
shift
TESTS_FILE=$1
shift
MODULES=$*

mkdir -p $TESTS_DIR
echo  `date` >$TESTS_FILE 
total_failures=0
for module in $MODULES ; do 
	echo "- run tests for $module" 
	cd $BUILD_DIR/$module 
	for ftest in `ls *.exe` ; do 
		echo "- $ftest" 
		./$ftest --gtest_output=xml:$TESTS_DIR/test_tmp.txt
		failed=`xpath -q -e "string(//testsuites/@failures)" $TESTS_DIR/test_tmp.txt 2>/dev/null` 
		total_failures=`expr $total_failures + $failed`
		cat $TESTS_DIR/test_tmp.txt >> $TESTS_FILE 
	done 
	cd $PROJECT_DIR 
done

echo "RESULTS in $TESTS_FILE"
echo "total failures=$total_failures"
