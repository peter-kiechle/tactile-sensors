#!/bin/bash

# Encode single *.png file to H.264 in a MP4 container using ffmpeg

# Example usage: ./create_video.sh animation/ foam_ball.dsa_ foam_ball_video

# $1: Path to image file ~/animation/foam_ball_frame_00000.png
# $2: Output filename without extention, e.g. foam_ball_video

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters!"
	echo "Example usage: ./create_video_loop.sh ~/animation/foam_ball_frame_00000.png still_video_foam_ball"
	exit
fi

profile=$1
profile_directory=${profile%/*} # Directory name 

# x264 high quality profile
ffmpeg -loop 1 -t 00:00:30 -i $profile -s:v 512x512 -c:v libx264 -profile:v high -crf 18 -pix_fmt yuv420p -r 30 -an $profile_directory"/"$2.mp4


# Lossless profile for use with green-screen overlays
#ffmpeg -loop 1 -t 00:00:30 -i $profile -s:v 512x512 -c:v libx264 -profile:v high444 -crf 0 -preset ultrafast -r 30 -an $profile_directory"/"$2.mp4

