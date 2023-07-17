#!/bin/bash

if [ -z "$1" ]; then
    exit 1
fi

git add *
# commit to cluster+recon branch
git commit -m "$1"

git push
pip install git+https://github.com/bycycle-tools/bycycle.git@cluster+recon --upgrade