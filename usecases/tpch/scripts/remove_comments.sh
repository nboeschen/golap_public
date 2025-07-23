#!/bin/bash

# this script removes the comments in already created tpch csv files.
# this understandably takes forever.
# instead of running this script, consider patching the dbgen source files by commenting out all "PR_VSTR_LAST"
# and running the generating executable again

if [[ $# -eq 0 ]] ; then
    echo "No csv folder given"
    echo "Usage: <script> <folder_containing_csvs>"
    exit 1
fi

cd $1

shopt -s nullglob
for file in $1/*.csv; do
    echo "Processing $file"
    mv $file $(basename "$file" .csv)_comments.csv
    rev $(basename "$file" .csv)_comments.csv | cut -d "|" -f 3- | rev > "$file"
    rm $(basename "$file" .csv)_comments.csv
done

# remove trailing field separator
# shopt -s nullglob
# for file in $1/*.tbl; do
#     echo "Processing $file"
#     sed -i 's/|$//' $file
# done
