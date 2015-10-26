#!/bin/bash

PWD="`pwd`"

if [ -n "$1" ]; then
   if [[ "$1" = /* ]]; # Absolute path
      then filename=$1
   else # Relative path: Add CWD
      filename="$PWD/$1"
   fi
   export DSA_PROFILE_NAME=$filename
fi

# cd to script's location in order to get relative library paths right
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # Directory of *this* file
cd $DIR &&

python PyDSA-Konqueror.py

