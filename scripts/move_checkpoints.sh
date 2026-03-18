#!/bin/bash

# run this script form the 'checkpoints' directory

for D in */ ; do
    [ -L "${D%/}" ] && continue
    echo "$D"
    cd "$D"
    if [ ! -d "weights" ]; then
        mkdir -p weights
        mv *weights*.h5 weights/
    fi
    cd ..
done
