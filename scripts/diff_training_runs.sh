#!/bin/sh

# run this script form the 'checkpoints' directory

# alias td='~/utils/scripts/diff_training_runs.sh'
# td 202601061950  202601062011


git diff --no-index "./$1/source.py" "./$2/source.py"
#git diff --no-index --unified=0 "./$1/source.py" "./$2/source.py"
