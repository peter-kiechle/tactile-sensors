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
import matplotlib.pyplot as plt

import framemanager_python

# Force reloading of external library (convenient during active development)
reload(framemanager_python)



def find_nearest(array, value):
    idx = np.argmin(np.abs(array - value))
    return array[idx]

def find_nearest_idx(array, value):
    idx = np.argmin(np.abs(array - value))
    return idx





#profileName = os.path.abspath("foam_ball.dsa")
#profileName = os.path.abspath("foam_ball_006077-006222.dsa")
profileName = os.path.abspath("hockey_ball.dsa")
#profileName = os.path.abspath("hockey_ball_000651-000810.dsa")

frameManager = framemanager_python.FrameManagerWrapper()
features = framemanager_python.FeatureExtractionWrapper(frameManager)

frameManager.load_profile(profileName);

#frameManager.set_filter_none()
#frameManager.set_filter_median(1, True)


########################
# Tactile Sensor Frames
########################
numTSFrames = frameManager.get_tsframe_count();

max_matrix_1 = frameManager.get_max_matrix_list(1)
max_matrix_5 = frameManager.get_max_matrix_list(5)

timestamps_tsframe_raw = frameManager.get_tsframe_timestamp_list()
timestamps_tsframe = (timestamps_tsframe_raw-timestamps_tsframe_raw[0]) / 1000.0 # Relative timestamps in seconds



###############
# Joint Angles
###############
# theta 0:  Rotational axis (Finger 0 + 2)
# theta 1:  Finger 0 proximal
# theta 2:  Finger 0 distal
# theta 3:  Finger 1 proximal
# theta 4:  Finger 1 distal
# theta 5:  Finger 2 proximal
# theta 6:  Finger 2 distal

numAngles = frameManager.get_jointangle_frame_count()
jointangles_list = frameManager.get_jointangle_frame_list()
timestamps_jointangles_raw = frameManager.get_jointangle_frame_timestamp_list()

# Relative timestamps in seconds
timestamps_jointangles = (timestamps_jointangles_raw-timestamps_jointangles_raw[0]) / 1000.0 

# Delete flawed measurements
#jointangles_list = np.delete(jointangles_list, (314), axis=0)
#timestamps_jointangles = np.delete(timestamps_jointangles, (314), axis=0)



###########
# Features
###########


# Compute miniball for each tactile sensor frame
minimal_bounding_spheres_ts = np.empty([numTSFrames, 4])
minimal_bounding_spheres_ts.fill(None)
for frameID in xrange(0, numTSFrames):    
    if (max_matrix_1[frameID] > 0.0 and max_matrix_5[frameID] > 0.0) :
        theta = frameManager.get_corresponding_jointangles(frameID) # Corresponding joint angles
        minimal_bounding_spheres_ts[frameID] = features.compute_minimal_bounding_sphere_centroid(frameID, theta)
                                                                                
'''                                                                 
# Compute miniball for each joint angle frame                                                                         
minimal_bounding_spheres_ja = np.empty([timestamps_jointangles.size, 4])
minimal_bounding_spheres_ja.fill(None)

for idx, theta in enumerate(jointangles_list):
    jointAngleTimestamp = timestamps_jointangles_raw[idx]
    TSframeID = find_nearest_idx(timestamps_tsframe_raw, jointAngleTimestamp) # Corresponding tactile sensor frame
    if (max_matrix_1[TSframeID] > 0.0 and max_matrix_5[TSframeID] > 0.0) :
        minimal_bounding_spheres_ja[TSframeID] = features.compute_minimal_bounding_sphere(TSframeID, theta)
'''
                                                                                 



#############
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


#---------------------------------------------------------
ax1 = plt.subplot(2,1,1)
#ax1.plot(timestamps_tsframe, averages_matrix_1, label="Average Matrix 1")
ax1.plot(max_matrix_1, label="Max matrix 1")
ax1.plot(max_matrix_5, label="Max matrix 5")

ax1.set_xlabel("\# Frames")
ax1.set_ylabel(r"Raw Sensor Value", rotation=90)
ax1.legend()
ax1.legend(loc = 'lower right')

ax1.set_xlim([0, numTSFrames-1])

#---------------------------------------------------------
# Miniball for each tactile sensor frame
ax2 = plt.subplot(2,1,2, sharex=ax1)
ax2.plot(minimal_bounding_spheres_ts[:,3], marker='.', label="Miniball Radius (corresponding to tactile frames)")

ax2.set_xlabel("\# Frames")
ax2.set_ylabel(r"Distance [mm]", rotation=90)
ax2.legend()
ax2.legend(loc = 'lower right')


#---------------------------------------------------------
'''
# Miniball for each joint angle frame  
ax3 = plt.subplot(3,1,3)
ax3.plot(timestamps_tsframe, minimal_bounding_spheres_ts[:,3], label="Miniball Radius (corresponding to tactile frames)")
ax3.plot(timestamps_jointangles, minimal_bounding_spheres_ja[:,3], label="Miniball Radius (corresponding to joint angles)")

ax3.legend()
ax3.legend(loc = 'lower right')
ax3.set_xlabel("Time [s]")
ax3.set_ylabel(r"Distance [mm]", rotation=90)
'''
#---------------------------------------------------------

#axes = plt.gca()
#axes.set_xlim([0, numTSFrames-1])

fig.tight_layout()
#plt.show()

plotname = "grasp_miniball"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf