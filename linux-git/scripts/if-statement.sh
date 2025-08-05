#!/bin/sh

if [ "$#" -ne 0 ]; then
    echo "You passed $# arguments"
else
    echo "You didn't pass any arguments"
fi
