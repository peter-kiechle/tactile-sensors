#!/bin/bash

# Encode sequence of *.png files to H.264 in a MP4 container using ffmpeg

# $1: Path and image prefix without consecutive numbers, e.g.  ~/animation/foam_ball_frame_
# $2: Output filename without extention, e.g. foam_ball_video

start_frame=42
source_fps=30

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters!"
	echo "Example usage: ./create_video.sh ~/animation/foam_ball_frame_ video_foam_ball"
	exit
fi

profile=$1
profile_directory=${profile%/*} # Directory name 
#profile_name=${profile##*/} # With extension
#profile_base_name=${profile_name%%.*} # Without extension

# x264 high quality profile
ffmpeg -r $source_fps -start_number $start_frame -i $profile%05d.png -s:v 512x512 -c:v libx264 -profile:v high -crf 18 -pix_fmt yuv420p -r 30 $profile_directory"/"$2.mp4

# Lossless profile for use with green-screen overlays
#ffmpeg -r $source_fps -start_number $start_frame -i $profile%05d.png -s:v 512x512 -c:v libx264 -profile:v high444 -crf 0 -preset ultrafast -r 30 $profile_directory"/"$2.mp4 
