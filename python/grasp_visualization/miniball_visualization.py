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
    idx = (np.abs(array-value)).argmin()
    return idx

                                                                           
def loadGrasp(profileName):
   frameManager = framemanager_python.FrameManagerWrapper()
   features = framemanager_python.FeatureExtractionWrapper(frameManager)
   frameManager.load_profile(profileName);

   ########################
   # Tactile Sensor Frames
   ########################
   numTSFrames = frameManager.get_tsframe_count();
   max_matrix_1 = frameManager.get_max_matrix_list(1)
   max_matrix_5 = frameManager.get_max_matrix_list(5)
   timestamps_tsframe_raw = frameManager.get_tsframe_timestamp_list()
   timestamps_tsframe = (timestamps_tsframe_raw-timestamps_tsframe_raw[0]) / 1000.0 # Relative timestamps in seconds

   ###########
   # Features
   ###########
   # Compute minibal for each tactile sensor frame
   minimal_bounding_spheres = np.empty([numTSFrames, 4])
   minimal_bounding_spheres.fill(None)

   for frameID in xrange(0, numTSFrames):    
      if (max_matrix_1[frameID] > 0.0 and max_matrix_5[frameID] > 0.0) :
         theta = frameManager.get_corresponding_jointangles(frameID) # Corresponding joint angles
         minimal_bounding_spheres[frameID] = features.compute_minimal_bounding_sphere_centroid(frameID, theta)
    
   return timestamps_tsframe, 2*minimal_bounding_spheres[:,3]



x_hockey, y_hockey = loadGrasp("hockey_ball_000651-000810.dsa")
x_foam, y_foam = loadGrasp("foam_ball_006077-006222.dsa")
        
                                                               

# Trim data right in the middle
# Hockey ball
start_time = find_nearest(x_hockey, 1.5)
stop_time = find_nearest(x_hockey, 4.2)
start_idx = find_nearest_idx(x_hockey, start_time) 
stop_idx = find_nearest_idx(x_hockey, stop_time) 
x_hockey = np.delete(x_hockey, np.s_[start_idx:stop_idx], axis=0)
y_hockey = np.delete(y_hockey, np.s_[start_idx:stop_idx], axis=0)
x_hockey[start_idx:] -= stop_time-start_time

# Foam ball
start_time = find_nearest(x_foam, 1.5)
stop_time = find_nearest(x_foam, 3.5)
start_idx = find_nearest_idx(x_foam, start_time) 
stop_idx = find_nearest_idx(x_foam, stop_time) 
x_foam = np.delete(x_foam, np.s_[start_idx:stop_idx], axis=0)
y_foam = np.delete(y_foam, np.s_[start_idx:stop_idx], axis=0)
x_foam[start_idx:] -= stop_time-start_time



#############
# Plotting
############

text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0

size_factor = 1.0
figure_width = size_factor*text_width
figure_height = (figure_width / golden_ratio)
#figure_height = 1.3 * figure_width
figure_size = [figure_width, figure_height]

config.load_config_medium()

#---------------------------------------------------------

fig = plt.figure(figsize=figure_size, dpi=100)
ax = fig.add_subplot(111)


ax.plot(x_hockey, y_hockey, linestyle="-", color=config.UIBK_orange, alpha=1.0, label="Grasping a hockey ball",
         marker='o', markeredgewidth=0.75, markersize=3.0, markeredgecolor=config.UIBK_orange, markerfacecolor=[1.0, 1.0, 1.0] )

ax.plot(x_foam, y_foam, linestyle="-", color=config.UIBK_blue, alpha=1.0, label="Grasping a foam ball",
         marker='s', markeredgewidth=0.75, markersize=3.0, markeredgecolor=config.UIBK_blue, markerfacecolor=[1.0, 1.0, 1.0] )

ax.set_xlabel("Time [s]")
ax.set_ylabel(r"Miniball Diameter [mm]", rotation=90)

ax.set_xlim([0, 2.5])
#ax.set_ylim([0, 850])
#ax.set_ylim([0, 1.1*ys2.max()])

# Legend
ax.legend(loc = 'upper center', fancybox=True, shadow=False, framealpha=1.0)

fig.tight_layout()
#plt.show()

plotname = "miniball_visualization"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf