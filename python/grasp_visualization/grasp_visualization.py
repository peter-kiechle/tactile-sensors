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
#reload(framemanager_python)




#profileName = os.path.abspath("foam_ball.dsa")
#profileName = os.path.abspath("foam_ball_006077-006222.dsa")
profileName = os.path.abspath("hockey_ball.dsa")
#profileName = os.path.abspath("hockey_ball_000651-000810.dsa")

frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileName);



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
corresponding_jointangles_list = frameManager.get_corresponding_jointangles_list()
timestamps_jointangles = frameManager.get_jointangle_frame_timestamp_list()
# Relative timestamps in seconds
timestamps_jointangles = (timestamps_jointangles-timestamps_jointangles[0]) / 1000.0 



'''
###############
# Temperatures
###############
# Temperatures 0-6: Close to corresponding axis motors
# Temperature 7: FPGA
# Temperature 8: Printed circuit board

numTemperatures = frameManager.get_temperature_frame_count()
temperature_frame_list = frameManager.get_temperature_frame_list()
corresponding_temperature_frame = frameManager.get_corresponding_temperatures_list()
timestamps_temperatures = frameManager.get_temperature_frame_timestamp_list()
# Relative timestamps in seconds
timestamps_temperatures = (timestamps_temperatures-timestamps_temperatures[0]) / 1000.0 
'''


#######################
# Tactile Sensor Frame
#######################
numTSFrames = frameManager.get_tsframe_count();

averages_matrix_1 = frameManager.get_average_matrix_list(1)
averages_matrix_5 = frameManager.get_average_matrix_list(5)

max_matrix_1 = frameManager.get_max_matrix_list(1)
max_matrix_5 = frameManager.get_max_matrix_list(5)

timestamps_tsframe = frameManager.get_tsframe_timestamp_list()
timestamps_tsframe = (timestamps_tsframe-timestamps_tsframe[0]) / 1000.0 # Relative timestamps in seconds



###########
# Features
###########

first_contact_matrix1 = np.where(max_matrix_1 > 0.0)[0][0]
first_contact_matrix5 = np.where(max_matrix_5 > 0.0)[0][0]

first_contact = max(first_contact_matrix1, first_contact_matrix5)
final_contact = numTSFrames # Assuming profile ends with active contacts

minimal_bounding_spheres = np.empty([numTSFrames, 4])
minimal_bounding_spheres.fill(None)

features = framemanager_python.FeatureExtractionWrapper(frameManager)

  
for frameID in xrange(0, numTSFrames):    
    if (max_matrix_1[frameID] > 0.0 and max_matrix_5[frameID] > 0.0) :
        theta = frameManager.get_corresponding_jointangles(frameID) # Corresponding joint angles
        #minimal_bounding_spheres[frameID] = features.compute_minimal_bounding_sphere(frameID, theta)
        minimal_bounding_spheres[frameID] = features.compute_minimal_bounding_sphere_centroid(frameID, theta)

# Note, for a more robust version, compute the centroids of a key frame and 
# stick to that coordinates using compute_minimal_bounding_sphere_points(), see train_classifier.py


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

#------------
# Matrix max
#------------
ax1 = plt.subplot(3,1,1)
ax1.plot(timestamps_tsframe, max_matrix_1, label="Max matrix 1")
ax1.plot(timestamps_tsframe, max_matrix_5, label="Max matrix 5")

ax1.legend(loc = 'lower right')
ax1.set_ylabel(r"Raw Sensor Value", rotation=90)


#---------------
# Joint angles
#---------------
ax2 = plt.subplot(3,1,2, sharex=ax1)
#ax2.plot(timestamps_jointangles, jointangles_list[:,1], label=r"$\theta_1$")
#ax2.plot(timestamps_jointangles, jointangles_list[:,2], label=r"$\theta_2$")
#ax2.plot(timestamps_jointangles, jointangles_list[:,5], label=r"$\theta_5$")
#ax2.plot(timestamps_jointangles, jointangles_list[:,6], label=r"$\theta_6$")

ax2.plot(timestamps_tsframe, corresponding_jointangles_list[:,1], label=r"$\theta_1$")
ax2.plot(timestamps_tsframe, corresponding_jointangles_list[:,2], label=r"$\theta_2$")
ax2.plot(timestamps_tsframe, corresponding_jointangles_list[:,5], label=r"$\theta_5$")
ax2.plot(timestamps_tsframe, corresponding_jointangles_list[:,6], label=r"$\theta_6$")

ax2.legend(loc = 'lower right')
ax2.set_ylabel(r"Joint Angle [$^\circ$]", rotation=90)


#-------------------------
# Minimal bounding sphere
#-------------------------
ax3 = plt.subplot(3,1,3, sharex=ax1)
ax3.plot(timestamps_tsframe, minimal_bounding_spheres[:,3], label="Miniball Radius")

ax3.legend(loc = 'lower right')
ax3.set_xlabel("Time [s]")
ax3.set_ylabel(r"Distance [mm]", rotation=90)

'''
#-------------
# Temperature
#-------------
ax4 = plt.subplot(4,1,4, sharex=ax1)
ax4.plot(timestamps_temperatures, temperature_frame_list[:,1], label="Temperature $\vartheta_1$")
ax4.plot(timestamps_temperatures, temperature_frame_list[:,5], label="Temperature $\vartheta_5$")

#ax4.legend(loc = 'lower right')
#ax4.set_xlabel("Time [s]")
#ax4.set_ylabel(r"Temperature [$\,^{\circ}\mathrm{C}$]", rotation=90)
'''

axes = plt.gca()
axes.set_xlim([0, timestamps_tsframe[-1]])

plt.show()

plotname = "grasp_visualization"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
#fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
