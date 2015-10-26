#!/bin/bash

# Extract digits and rename file since ffmpeg is picky with sequence filenames (max 5 digits, start_number option)

# $1: Path of *.png files

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters!"
	echo "Example usage: ./renaming_sequence.sh ~/animation/"
	exit
fi

#frameID=1
image_directory=${1%/} # Remove single trailing slash

for image in $(ls $image_directory"/"*.png) ; do
	
	image_name=${image##*/} # With extension
	
	# Groups are: filename, digits, extension
	filename=$(echo "$image_name"  | sed "s/\([^0-9]*\)\([0-9]\+\)\(.*\)/\1/" ); # First group
	digits=$(echo "$image_name"    | sed "s/\([^0-9]*\)\([0-9]\+\)\(.*\)/\2/" ); # Second group
	extension=$(echo "$image_name" | sed "s/\([^0-9]*\)\([0-9]\+\)\(.*\)/\3/" ); # Third group
	
	# Remove leading zeros as printf() expects octal otherwise
	digits=$(echo $digits | sed 's/^0*//')
	
	image_renamed=$(printf "%s%05d%s" $filename $digits $extension) 
	image_renamed=$image_directory"/"$image_renamed
	
	#echo $image_renamed
	mv -v $image $image_renamed
	
	#let frameID=frameID+1
done