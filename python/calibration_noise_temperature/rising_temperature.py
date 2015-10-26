# -*- coding: utf-8 -*-

import os, sys
print("CWD: " + os.getcwd() )

# Load configuration file before pyplot
config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)
import configuration as config

# Library path
lib_path = os.path.abspath('../../lib')
sys.path.append(lib_path)
import framemanager_python
#reload(framemanager_python) # Force reloading of external library (convenient during active development)


import matplotlib.pyplot as plt
import numpy as np
#import brewer2mpl
 
#----------------------------------------
# Load raw sensor and temperature values
#----------------------------------------

#data = np.genfromtxt("noise_rising_temperature_threshold_calibration_temperature.dat", dtype=float, delimiter=';', names=True)
#names = data.dtype.names[1:]

profileName = os.path.abspath("temperature-noise.dsa")
frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileName);



###############################
# Characteristic sensor values
###############################
sensor_values = np.column_stack((frameManager.get_max_matrix_list(0),
                                 frameManager.get_max_matrix_list(1),
                                 frameManager.get_max_matrix_list(2),
                                 frameManager.get_max_matrix_list(3),
                                 frameManager.get_max_matrix_list(4),
                                 frameManager.get_max_matrix_list(5) ))
                                 
# Time stamps of tactile sensors
timestamps_ts = frameManager.get_tsframe_timestamp_list()
timestamps_ts = (timestamps_ts-timestamps_ts[0]) / 1000.0 # Relative timestamps in seconds


###############
# Temperatures
###############

# Temperatures 0-6: Close to corresponding motors
# Temperature 7: FPGA
# Temperature 8: Printed circuit board
numTemperatures = frameManager.get_temperature_frame_count()
temperature_frames_full = frameManager.get_temperature_frame_list()
# slice of relevant axis temperatures
# i.e. (axis 1 -> matrix 0), (axis 2 -> matrix 1) ... (axis 6 -> matrix 5)
temperature_frames = temperature_frames_full[:,1:7] 

# Time stamps of temperature readings
timestamps_temperature = frameManager.get_temperature_frame_timestamp_list()
timestamps_temperature = (timestamps_temperature-timestamps_temperature[0]) / 1000.0 # Relative timestamps in seconds
timestamps_temperature /= 60.0



###########
# Plotting
###########

text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0
size_factor = 1.0
figure_width = size_factor*text_width
figure_height = (figure_width / golden_ratio)
#figure_height = 1.3 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_medium()



'''
# Some Categorical colors
colors = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), # Blue
          (1.0, 0.4980392156862745, 0.054901960784313725), # Orange
          (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), # Green
          (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), # Red
          (0.5803921568627451, 0.403921568627451, 0.7411764705882353), # Purple
          (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), # Brown
          (0.5, 0.5, 0.5), # Gray
          (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), # Greenish/yellow
          (0.09019607843137255, 0.7450980392156863, 0.8117647058823529), # Aquamarine
          (0.8901960784313725, 0.4666666666666667, 0.7607843137254902)]  # Pink
'''

'''
# Custom categorical colors
colors = [(0.8392156862745098, 0.1568627450980392, 0.1568627450980392), # Red
          (0.0, 0.17647058823529413, 0.4392156862745098),               # Blue 1
          (1.0, 0.5, 0.0),                                              # Orange 1
          (0.3411764705882353, 0.4549019607843137, 0.6274509803921569), # Blue 2
          (1.0, 0.6666666666666666, 0.3411764705882353),                # Orange 2
          (0.6705882352941176, 0.7254901960784313, 0.8117647058823529), # Blue 3
          (1.0, 0.8313725490196079, 0.6705882352941176),                # Orange 3
          (0.5, 0.5, 0.5)]                                              # Gray
'''

#(0.50196078431, 0.33725490196, 0.21568627451), # Brown
colors = [(0.8392156862745098, 0.1568627450980392, 0.1568627450980392), # Red 
          config.alphablend(config.UIBK_orange, 0.33),
          config.alphablend(config.UIBK_blue, 0.33),    
          config.alphablend(config.UIBK_orange, 0.66),
          config.alphablend(config.UIBK_blue, 0.66),
          config.UIBK_orange, 
          config.UIBK_blue,
          (0.1, 0.1, 0.1)]                                              

     

     
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figure_size, dpi=100)

temperature_sensors = ["Axis 0 (base)", "Axis 1 (proximal)", "Axis 2 (distal)", "Axis 3 (proximal)", "Axis 4 (distal)", "Axis 5 (proximal)", "Axis 6 (distal)", "FPGA"]


plotting_order = np.array([0,1,3,5,2,4,6,7])
for entry in plotting_order:
    i = entry
    sensor = temperature_sensors[i]
    ax.plot(timestamps_temperature, temperature_frames_full[:,i], ls='-', lw=2, color=colors[i], alpha=1.0, label=sensor)
    
    
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=1.0)
ax.text(1500, 37, r"Distal axis motors", 
        horizontalalignment='left',
        verticalalignment='center',
        fontsize='medium',
        bbox=bbox_props)

ax.text(2000, 46, r"Rotational axis", 
        horizontalalignment='left',
        verticalalignment='center',
        fontsize='medium',
        bbox=bbox_props)
        
ax.text(2500, 54, r"Proximal axis motors", 
        horizontalalignment='left',
        verticalalignment='center',
        fontsize='medium',
        bbox=bbox_props)


#ax.set_title(r"Rising temperature under static load (0.4 Ampere)")
ax.set_xlabel("Time [min]")
ax.set_ylabel("Temperature [$\,^{\circ}\mathrm{C}$]", rotation=90)
ax.legend(loc = 'upper left', fancybox=True, shadow=False, framealpha=1.0)

ax.set_xlim([0,65])
ax.set_ylim([20, 70])


fig.tight_layout()
#plt.show() 

plotname = "rising_temperature"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
