#!/bin/sh

func() {
    if [ "$#" -gt 0 ]; then
	echo "Function provided $# arguments:"
	for i in $@
	do
	    echo "\t| ${i}"
	done
    else
	echo "Function provided no arguments."
    fi
}

func
func a b c
func 1 2
func
