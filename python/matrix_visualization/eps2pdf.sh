#!/bin/bash
f=$1
epstopdf $f
basename=${f%%.*}

pdfcrop "$basename.pdf"
rm "$basename.pdf"
mv "$basename-crop.pdf" "$basename.pdf"

