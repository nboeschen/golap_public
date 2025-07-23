#!/bin/sh
cd "$(dirname $(readlink -f "$0"))/.."

git grep -rli "#ifdef WITH_GDSASYNC" . | grep -v "create_archive.sh" | xargs -i@ sed -i "/#ifdef WITH_GDSASYNC/,/#endif \/\/WITH_GDSASYNC/d" @
git grep -rli "#ifdef WITH_AWS" . | grep -v "create_archive.sh" | xargs -i@ sed -i "/#ifdef WITH_AWS/,/#endif \/\/WITH_AWS/d" @

stashName=`git stash create`;
git archive --format=zip $stashName -o "golap_src.zip"


# only include this when there are no untracked files or uncomitted changes
git checkout .
