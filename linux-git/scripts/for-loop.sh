#!/bin/sh

if [ "$#" -ge 1 ]; then
    for i in $(seq 1 $(($1)))
    do
	line=""
	for j in $(seq 1 $i)
	do
	     line="${line} *"
	done
	echo "${line}"
    done
else
    echo "Enter at least one argument"
fi
