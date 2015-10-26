#!/bin/bash

# Convert colored pdf-file to grayscale using ghostscipt

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters!"
    echo "Example usage: ./convert_black_and_white.sh ~/masterthesis/masterthesis.pdf"
    exit
fi

FILE=$1
FILE_DIR=$(dirname "${FILE}") # Directory name 
FILE_NAME="${FILE##*/}" # With extension
FILE_BASENAME="${FILE_NAME%%.*}" # Without extension
#echo "FILE: $FILE"
#echo "FILE_DIR: $FILE_DIR"
#echo "FILE_NAME: $FILE_NAME"
#echo "FILE_BASENAME: $FILE_BASENAME"

OUTFILE=$FILE_DIR"/"$FILE_BASENAME"_black_and_white.pdf"

gs \
  -o $OUTFILE \
  -sDEVICE=pdfwrite \
  -dPDFSETTINGS=/prepress \
  -sColorConversionStrategy=Gray \
  -sColorConversionStrategyForImages=Gray \
  -sProcessColorModel=DeviceGray \
  -dCompatibilityLevel=1.4 \
   $FILE  