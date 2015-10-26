# -*- coding: utf-8 -*-

import os, sys
print("CWD: " + os.getcwd() )

config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)
lib_path = os.path.abspath('../../lib')
sys.path.append(lib_path)

# Load configuration file (before pyplot)
import configuration as config


import numpy as np
import scipy.ndimage as ndi
import cv2
import matplotlib.pyplot as plt


import framemanager_python

# Force reloading of external library (convenient during active development)
reload(framemanager_python)


# Taken from http://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array
# Author: Joe Kington
def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx








frameManager = framemanager_python.FrameManagerWrapper()
features = framemanager_python.FeatureExtractionWrapper(frameManager)

profileName = os.path.abspath("some_steps.dsa")
#profileName = os.path.abspath("single_step.dsa")
frameManager.load_profile(profileName);

frameManager.set_filter_none()
#frameManager.set_filter_median(1, True)



numTSFrames = frameManager.get_tsframe_count();
starttime = frameManager.get_tsframe_timestamp(0)
stoptime = frameManager.get_tsframe_timestamp(numTSFrames)

max_matrix_1 = frameManager.get_max_matrix_list(1)
max_matrix_5 = frameManager.get_max_matrix_list(5)

timestamps = frameManager.get_tsframe_timestamp_list()
timestamps = (timestamps-timestamps[0]) / 1000.0 # Relative timestamps in seconds

corresponding_jointangles = frameManager.get_corresponding_jointangles_list()


# Compute minibal for each tactile sensor frame
miniballs = np.empty([numTSFrames, 4])
miniballs.fill(None)

for frameID in xrange(0, numTSFrames):    
    if (max_matrix_1[frameID] > 0.0 and max_matrix_5[frameID] > 0.0) :
        theta = corresponding_jointangles[frameID]
        miniballs[frameID] = features.compute_minimal_bounding_sphere_centroid(frameID, theta)
radius = miniballs[:,3]


#---------------------------------
# Simple step detection algorithm
#---------------------------------
# Find all non-zero sequences
# Throw small sequencs away. Actual grasps are remaining
# For more elaborated methods: http://en.wikipedia.org/wiki/Step_detection

thresh_sequence = 10 # Minimum length of a sequence to be considered a "grasp"
grasp_begin = []
grasp_end = []

#radius_mask = np.logical_not(np.isnan(radius)).astype(int)
for start, stop in contiguous_regions( np.logical_not(np.isnan(radius)) ):
    if (stop-start) > thresh_sequence:
        grasp_begin.append([start, radius[start]])
        grasp_end.append([stop-1, radius[stop-1]])




############
# Plotting
############
brewer_red = [0.89411765, 0.10196078, 0.10980392]
brewer_blue = [0.21568627, 0.49411765, 0.72156863]
brewer_green = [0.30196078, 0.68627451, 0.29019608]

text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0

size_factor = 1.0
figure_width = size_factor*text_width
#figure_height = (figure_width / golden_ratio)
figure_height = 1.3 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_medium()


fig = plt.figure(figsize=figure_size)

# Axis 1
ax1 = fig.add_subplot(2,1,1)

ax1.plot(max_matrix_1, "-", color=brewer_blue, label="Max Matrix 1")
ax1.plot(max_matrix_5, "-", color=brewer_blue, alpha=0.5, label="Max Matrix 5")


# Grasp begin / end
#ax1.plot([p[0] for p in grasp_begin], [p[1] for p in grasp_begin], "o", markersize=8, color="green", label="Grasp begin")
#ax1.plot([p[0] for p in grasp_end], [p[1] for p in grasp_end], "o", markersize=8, color="red", label="Grasp end")

for frameID in [p[0] for p in grasp_begin]:
    ax1.axvline(frameID, color=brewer_green, linestyle='solid')
for frameID in [p[0] for p in grasp_end]:    
    ax1.axvline(frameID, color=brewer_red, linestyle='solid')

ax1.set_xlim([0,60])
#ax1.set_ylim([0, 1.2*np.max(max_matrix_5)])


ax1.set_xlabel("\# Frames")
ax1.set_ylabel("Raw Sensor Value", rotation=90)
ax1.set_title("Step detection by finding long non-zero sequences in miniball radius", y=1.10)

'''
# Second axis for time
ax1_time = ax1.twiny()
dummy = ax1_time.plot(timestamps, np.ones([timestamps.size]))
dummy.pop(0).remove()
ax1_time.set_xlabel("Time [s]")
'''

ax1.legend()
ax1.legend(loc = 'upper left')


# Axis 2
ax2 = fig.add_subplot(2,1,2, sharex=ax1)

ax2.plot(radius, color=[0.3, 0.3, 0.3, 1.0], label="Miniball radius")

# Grasp begin / end
for frameID in [p[0] for p in grasp_begin]:
    ax2.axvline(frameID, color='green', linestyle='solid')
for frameID in [p[0] for p in grasp_end]:    
    ax2.axvline(frameID, color='red', linestyle='solid')


ax2.set_xlabel(r"\n# Frames")
ax2.set_ylabel("Distance [mm]", rotation=90)
ax2.legend()
ax2.legend(loc = 'upper left')


fig.tight_layout()
#plt.show() 

plotname = "step_detection_miniball"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
#fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()


