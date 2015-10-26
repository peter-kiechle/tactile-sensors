#!/bin/bash

# Rotate and resize all *.png files in directory

# $1: Path of *.png files

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters!"
	echo "Example usage: ./resize_and_rotate.sh ~/animation/"
	exit
fi

image_directory=${1%/} # Remove single trailing slash

for image in `ls $image_directory"/"*.png`; do
	image_name=${image##*/} # With extension
	image_processed=$image_directory"/processed_"$image_name
	image_processed_green=$image_directory"/processed_green_"$image_name
	
	echo "Rotating and resizing: "$image_name

	# Rotate
	convert $image -rotate 180 $image_processed
	
	# Remove background
	#convert $image_processed -alpha set -channel RGBA -fill none -opaque 'rgb(0,0,0)' $image_processed # Black
	convert $image_processed -alpha set -channel RGBA -fill none -opaque 'rgb(255,255,255)' $image_processed # White
	
	# Resize
	convert $image_processed -resize 768x768 $image_processed
	
	# Add green-screen
	# convert $image_processed -fuzz 30% -fill 'rgb(0,255,0)' -opaque 'rgb(0,0,0)' $image_processed
	# convert $image_processed -background 'rgb(0,255,0)' -flatten $image_processed_green

done

