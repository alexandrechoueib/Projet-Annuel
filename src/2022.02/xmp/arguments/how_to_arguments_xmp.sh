#!/bin/sh
echo "in bash $1"
exe=$1

echo "---------- use short options ----------"
$exe -b true -i -2 -n 3 -r 3.14 -x 12 -y one
echo "---------- use long options ----------"
$exe --boolean=true --integer=-2 --natural=3 --real=3.14 --integer_range=20 --option=four
