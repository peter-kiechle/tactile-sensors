# -*- coding: utf-8 -*-

import os, sys
print("CWD: " + os.getcwd() )
lib_path = os.path.abspath('../lib')
sys.path.append(lib_path)

import numpy as np
import matplotlib.pyplot as plt

import framemanager_python

# Force reloading of external library (convenient during active development)
#reload(framemanager_python)


profileName = os.path.abspath("reactive_grasping.dsa")
frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileName);
frameManager.set_filter_median(1, False)

numTSFrames = frameManager.get_tsframe_count();
starttime = frameManager.get_tsframe_timestamp(0)
stoptime = frameManager.get_tsframe_timestamp(numTSFrames)
frameID = 42


######################
# Access frames (copy)
######################
tsframe0 = np.copy( frameManager.get_tsframe(frameID, 0) );
tsframe1 = np.copy( frameManager.get_tsframe(frameID, 1) );
tsframe2 = np.copy( frameManager.get_tsframe(frameID, 2) );
tsframe3 = np.copy( frameManager.get_tsframe(frameID, 3) );
tsframe4 = np.copy( frameManager.get_tsframe(frameID, 4) );
tsframe5 = np.copy( frameManager.get_tsframe(frameID, 5) );

# It's possible to use vstack() to concatenate frames since proximal and distal frames do not differ in width
# tsframe_stacked = np.vstack([tsframe0, tsframe1, tsframe2, tsframe3, tsframe4, tsframe5])

# Normalize frames
tsframe0 /= max(1.0, frameManager.get_max_matrix(frameID, 0))
tsframe1 /= max(1.0, frameManager.get_max_matrix(frameID, 1))
tsframe2 /= max(1.0, frameManager.get_max_matrix(frameID, 2))
tsframe3 /= max(1.0, frameManager.get_max_matrix(frameID, 3))
tsframe4 /= max(1.0, frameManager.get_max_matrix(frameID, 4))
tsframe5 /= max(1.0, frameManager.get_max_matrix(frameID, 5))


########################
# Characteristic values
########################

averages = frameManager.get_average_frame_list()
averages_matrix_0 = frameManager.get_average_matrix_list(0)
averages_matrix_1 = frameManager.get_average_matrix_list(1)
averages_matrix_2 = frameManager.get_average_matrix_list(2)
averages_matrix_3 = frameManager.get_average_matrix_list(3)
averages_matrix_4 = frameManager.get_average_matrix_list(4)
averages_matrix_5 = frameManager.get_average_matrix_list(5)

mins = frameManager.get_min_frame_list()
min_matrix_0 = frameManager.get_min_matrix_list(0)
min_matrix_1 = frameManager.get_min_matrix_list(1)
min_matrix_2 = frameManager.get_min_matrix_list(2)
min_matrix_3 = frameManager.get_min_matrix_list(3)
min_matrix_4 = frameManager.get_min_matrix_list(4)
min_matrix_5 = frameManager.get_min_matrix_list(5)

maxs = frameManager.get_max_frame_list() # 3405
max_matrix_0 = frameManager.get_max_matrix_list(0)
max_matrix_1 = frameManager.get_max_matrix_list(1)
max_matrix_2 = frameManager.get_max_matrix_list(2)
max_matrix_3 = frameManager.get_max_matrix_list(3)
max_matrix_4 = frameManager.get_max_matrix_list(4)
max_matrix_5 = frameManager.get_max_matrix_list(5)

# Time stamps
timestamps = frameManager.get_tsframe_timestamp_list()
timestamps = (timestamps-timestamps[0]) / 1000.0 # Relative timestamps in seconds



###############
# Joint Angles
###############
# 0 : common base axis of finger 0 and 2
# 1 : proximal axis of finger 0
# 2 : distal axis of finger 0
# 3 : proximal axis of finger 1
# 4 : distal axis of finger 1
# 5 : proximal axis of finger 2
# 6 : distal axis of finger 2

angleID = 42
numAngles = frameManager.get_jointangle_frame_count()
jointangle_frame = frameManager.get_jointangle_frame(angleID)
jointangle_frame_list = frameManager.get_jointangle_frame_list()
jointangle_frame_timestamp = frameManager.get_jointangle_frame_timestamp(angleID)
jointangle_frame_timestamp_list = frameManager.get_jointangle_frame_timestamp_list()
corresponding_jointangle_frame = frameManager.get_corresponding_jointangles(frameID)


###############
# Temperatures
###############
# Temperatures 0-6: Close to corresponding motors
# Temperature 7: FPGA
# Temperature 8: Printed circuit board

tempID = 2
numTemperatures = frameManager.get_temperature_frame_count()
temperature_frame = frameManager.get_temperature_frame(tempID)
temperature_frame_list = frameManager.get_temperature_frame_list()
temperature_frame_timestamp = frameManager.get_temperature_frame_timestamp(tempID)
temperature_frame_timestamp_list = frameManager.get_temperature_frame_timestamp_list()
corresponding_temperature_frame = frameManager.get_corresponding_temperatures(frameID)


###################
# Compute Features
###################
features = framemanager_python.FeatureExtractionWrapper(frameManager)
std_dev = features.compute_standard_deviation(frameID, 1) # matrixID
moments = features.compute_chebyshev_moments(frameID, 1, 3) # matrixID, pmax
theta = frameManager.get_corresponding_jointangles(frameID) # Corresponding joint angles
minimal_bounding_sphere = features.compute_minimal_bounding_sphere(frameID, theta)


# Using lists
tsframe_list = frameManager.get_tsframe_list(frameID);
std_dev_list = features.compute_standard_deviation_list(frameID)
moments_list = features.compute_chebyshev_moments_list(frameID, 3)



############
# Plotting
############

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

dummy = ax2.plot(timestamps, max_matrix_1)
dummy.pop(0).remove()

ax1.plot(averages_matrix_0, label="Average Matrix 0")
ax1.plot(averages_matrix_1, label="Average Matrix 1")
ax1.plot(averages_matrix_2, label="Average Matrix 2")
ax1.plot(averages_matrix_3, label="Average Matrix 3")
ax1.plot(averages_matrix_4, label="Average Matrix 4")
ax1.plot(averages_matrix_5, label="Average Matrix 5")

#ax1.plot(min_matrix_1, label="Min Matrix 1")
#ax1.plot(min_matrix_3, label="Min Matrix 3")
#ax1.plot(min_matrix_5, label="Min Matrix 5")

#ax1.plot(max_matrix_1, label="Max Matrix 1")
#ax1.plot(max_matrix_3, label="Max Matrix 3")
#ax1.plot(max_matrix_5, label="Max Matrix 5")

ax1.legend(loc = 'lower right')
ax1.set_xlabel("# Frames")
ax2.set_xlabel("Time [s]")
ax1.set_ylabel("Raw Sensor Value", rotation=90)

plt.show()

