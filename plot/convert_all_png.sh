#!/bin/bash
set -e
cd "$(dirname $(readlink -f "$0"))"

mkdir -p png

for f in ./pdf/*.pdf; do
    filename="${f##*/}"
    filename="${filename%.pdf}"
    # echo "$filename"
    pdftoppm "$f" "./png/$filename" -png
done

