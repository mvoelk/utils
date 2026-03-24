#!/bin/sh

# run this script form the 'checkpoints' directory

# alias td='~/utils/scripts/diff_training_runs.sh'
# td 202601061950  202601062011

git diff --no-index --color --color-words=. --unified=1 "./$1/source.py" "./$2/source.py"
