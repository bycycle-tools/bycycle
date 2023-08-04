#!/bin/bash

if [ -z "$1" ]; then
    exit 1
fi

git add *
# commit to cluster+recon branch
git commit -m "$1"

git push
# pip install git+https://github.com/bycycle-tools/bycycle.git@cluster+recon
pip install -e .

# ryan's commands:
# python3 -m venv .env
# source .env/bin/activate
# git checkout -b nipype origin/nipype
# git stash
# git checkout -b nipype origin/nipype
# pip install -e .
# git stash pop
# git stash pop
# git status
# pip install -e .
# pip install scikit-learn
# pip install -U nbformat
# pip install jupyter

