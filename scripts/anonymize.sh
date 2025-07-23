#!/bin/bash

cd "$(dirname $(readlink -f "$0"))"

if [[ $# -eq 0 ]] ; then
    echo "No name/string to remove supplied"
    exit 1
elif [[ $# -eq 1 ]]; then
    git grep -rli "$1" ..
elif [[ $# -eq 2 ]]; then
    git grep -rli "$1" .. | xargs -i@ sed -i "s/$1/$2/g" @
fi

