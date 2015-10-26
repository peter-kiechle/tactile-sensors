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







profileName = os.path.abspath("some_steps.dsa")
frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileName);


numTSFrames = frameManager.get_tsframe_count();
starttime = frameManager.get_tsframe_timestamp(0)
stoptime = frameManager.get_tsframe_timestamp(numTSFrames)

max_matrix_1 = frameManager.get_max_matrix_list(1)
max_matrix_5 = frameManager.get_max_matrix_list(5)

# Time stamps
timestamps = frameManager.get_tsframe_timestamp_list()
timestamps = (timestamps-timestamps[0]) / 1000.0 # Relative timestamps in seconds



# Simple smoothing
#filtered_matrix_5 = ndi.filters.median_filter(max_matrix_5, size=5, mode='reflect')
#filtered_matrix_5 = ndi.uniform_filter1d(max_matrix_5, size=10, mode='reflect')

# Edge detection
sobel = ndi.sobel(max_matrix_5, mode='reflect')  
#laplace = ndi.laplace(max_matrix_5, mode='reflect') # Too sensitive to noise
#gaussian_laplace = ndi.filters.gaussian_laplace(max_matrix_5, sigma=1.0, mode='reflect')
#gaussian_gradient_magnitude = ndi.filters.gaussian_gradient_magnitude(max_matrix_5, sigma=1.0, mode='reflect')

max_matrix_5_cv = (max_matrix_5/(4096.0/255.0)).astype(np.uint8) # Scale [0..255], convert to CV_8U
canny = cv2.Canny(max_matrix_5_cv, 10, 20) # Hysteresis Thresholding: 
canny = canny.astype(np.float64) * (sobel.max()/255.0) # Scale to comparable scale


#---------------------------------
# Simple step detection algorithm
#---------------------------------
# Find all non-zero sequences
# Throw small sequencs away. Actual grasps are remaining
# For more elaborated methods: http://en.wikipedia.org/wiki/Step_detection

thresh_sequence = 10 # Minimum length of a sequence to be considered a "grasp"
grasp_begin = []
grasp_end = []
for start, stop in contiguous_regions(max_matrix_5 != 0):
    if (stop-start) > thresh_sequence:
        grasp_begin.append([start, max_matrix_5[start]])
        grasp_end.append([stop-1, max_matrix_5[stop-1]])







############
# Plotting
############
text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0

size_factor = 1.0
figure_width = size_factor*text_width
#figure_height = (figure_width / golden_ratio)
figure_height = 1.3 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_large()





fig = plt.figure(figsize=figure_size)

# Axis 1
ax1 = fig.add_subplot(2,1,1)

ax1.plot(max_matrix_5, "-", label="Max Matrix 5")
#ax1.plot(filtered_matrix_5, "-", marker="x", markersize=4, label="Median Matrix 5")

ax1.plot([p[0] for p in grasp_begin], [p[1] for p in grasp_begin], "o", markersize=8, color="green", label="Grasp begin")
ax1.plot([p[0] for p in grasp_end], [p[1] for p in grasp_end], "o", markersize=8, color="red", label="Grasp end")


#ax1.set_xlim([xmin,xmax])
ax1.set_ylim([0, 1.2*np.max(max_matrix_5)])
ax1.set_xlabel("# Frames")
ax1.set_ylabel("Raw Sensor Value", rotation=90)
ax1.set_title("Step detection by finding long non-zero sequences in tactile sensor readings", y=1.10)

# Second axis for time
ax1_time = ax1.twiny()
dummy = ax1_time.plot(timestamps, np.ones([timestamps.size]))
dummy.pop(0).remove()
ax1_time.set_xlabel("Time [s]")
ax1.legend(loc = 'upper left')




# Axis 2
ax2 = fig.add_subplot(2,1,2, sharex=ax1)

ax2.plot(sobel, label="Sobel")
#ax2.plot(laplace, label="Laplace")
#ax2.plot(gaussian_laplace, label="Gaussian laplace")
#ax2.plot(gaussian_gradient_magnitude, label="Gaussian gradient magnitude")
ax2.plot(canny, label="Canny")

ax2.set_xlabel("# Frames")
ax2.set_ylabel("Filtered", rotation=90)
ax2.legend(loc = 'lower left')


fig.tight_layout()
#plt.show() 


plotname = "step_detection_tactile_sensors"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
#fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()
