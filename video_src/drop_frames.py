# -*- coding: utf-8 -*-

# Removes specified number of images from directory as fair as possible.
# This might be necessary in case the video frame rate and the tactile sensor frame rate differ significantly.
# Expecting a directory containing a sequence of *.png files (5 digit sequence)
# e.g. egg_matrix_1_00257.png, egg_matrix_1_00258.png, egg_matrix_1_00259.png ...


import os, sys
import shutil # move

import re
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('filename', help='Any file of the sequence')
parser.add_argument('N', type=int, help='Number of frames to drop')
args = parser.parse_args()

filename = args.filename
num_drop = args.N

path = os.path.dirname(filename)
files = [f for f in os.listdir(path) if f.endswith('.png')]
files.sort()

basename = re.findall(r"(.*)(\d{5})(\.png)$", files[0])[0][0]
first_frame = int( re.findall(r".*(\d{5})\.png$", files[0])[0] )
last_frame = int( re.findall(r".*(\d{5})\.png$", files[-1])[0] )
num_frames = last_frame-first_frame+1

full_set = np.arange(first_frame, last_frame+1, 1) 
remaining_set = np.linspace(start=first_frame, stop=last_frame, num=num_frames-num_drop, endpoint=True, dtype=int)
candidates = np.setdiff1d(full_set, remaining_set)

# Create directory for dismissed frames
newdir = os.path.join(path, "dropped frames")
if not os.path.exists(newdir):
    os.makedirs(newdir)

# Move dismissed frames
for c in candidates:
    source = os.path.join(path, basename + "%05d" % c + ".png")
    destination = os.path.join(newdir, basename + "%05d" % c + ".png")
    #print source, destination
    shutil.move(source, destination)
    
print ("%d files moved!" % (args.N))
