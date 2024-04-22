#!/bin/bash

for D in */ ; do
    [ -L "${D%/}" ] && continue
    echo "$D"
    cd "$D"
    ls -1 weights*.h5 2> /dev/null | head -n -1 | grep -v "0\.h5" | xargs -i rm -v {}
    #ls -1 weights*.h5 2> /dev/null | head -n -1 | grep -v "00\.h5" | xargs -i rm -v {}
   cd ..
done
