#!/bin/bash

for f in *.png
do
   basename=${f%%.*}
   _png2eps.py $f
   _eps2pdf.sh $basename.eps
   rm "$basename.eps"
done
