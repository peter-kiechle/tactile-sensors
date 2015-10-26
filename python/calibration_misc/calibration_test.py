# -*- coding: utf-8 -*-

# Load configuration file before pyplot
import os, sys
config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)
import configuration as config

# Library path
import os, sys
print("CWD: " + os.getcwd() )
lib_path = os.path.abspath('../../lib')
sys.path.append(lib_path)

import matplotlib.pyplot as plt
import numpy as np

import framemanager_python
# Force reloading of external library (convenient during active development)
reload(framemanager_python)


def load_data_average(filename, matrixID):
    profileName = os.path.abspath(filename)
    frameManager = framemanager_python.FrameManagerWrapper()
    frameManager.load_profile(profileName);
    frameManager.set_filter_none()
    averages = frameManager.get_average_matrix_list(matrixID)
    timestamps = frameManager.get_tsframe_timestamp_list()
    timestamps = (timestamps-timestamps[0]) / 1000.0 # Relative timestamps in seconds
    return timestamps, averages
 
def load_smoothed(filename, matrixID):
    profileName = os.path.abspath(filename)
    frameManager = framemanager_python.FrameManagerWrapper()
    frameManager.load_profile(profileName);
    frameManager.set_filter_gaussian(1, 0.85)
    y = frameManager.get_average_matrix_list(matrixID)
    #y = frameManager.get_max_matrix_list(matrixID)
    
    timestamps = frameManager.get_tsframe_timestamp_list()
    timestamps = (timestamps-timestamps[0]) / 1000.0 # Relative timestamps in seconds
    return timestamps, y

   
def load_data_taxel(filename, matrixID, x, y):
    profileName = os.path.abspath(filename)
    frameManager = framemanager_python.FrameManagerWrapper()
    frameManager.load_profile(profileName);
    texel = frameManager.get_texel_list(matrixID, x, y)
    timestamps = frameManager.get_tsframe_timestamp_list()
    timestamps = (timestamps-timestamps[0]) / 1000.0 # Relative timestamps in seconds
    return timestamps, texel

def find_nearest_idx(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx





# Load Data
matrixID = 2

x,y = load_data_average("calibration_test_1000_3cm_cropped.dsa", matrixID)
x_s,y_s = load_smoothed("calibration_test_1000_3cm_cropped.dsa", matrixID)
x1,y1 = load_data_taxel("calibration_test_1000_3cm_cropped.dsa", matrixID, 4, 9)
x2,y2 = load_data_taxel("calibration_test_1000_3cm_cropped.dsa", matrixID, 5, 8)
x3,y3 = load_data_taxel("calibration_test_1000_3cm_cropped.dsa", matrixID, 4, 6)
x4,y4 = load_data_taxel("calibration_test_1000_3cm_cropped.dsa", matrixID, 5, 6)

# Trim data
start_idx = max(np.argmax(y!=0)-1, 0) # Index of first non-zero value (or 0)
start_time = x[start_idx]
duration = 4.1
stop_idx = find_nearest_idx(x, x[start_idx]+duration) 
#stop_idx = y.shape[0]

# TODO:
# Autodetect duration and/or stop_idx

x = x[start_idx:stop_idx]-start_time
y = y[start_idx:stop_idx]
x_s = x_s[start_idx:stop_idx]-start_time
y_s = y_s[start_idx:stop_idx]
#N = x.shape[0]

x1 = x1[start_idx:stop_idx]-start_time
y1 = y1[start_idx:stop_idx]
x2 = x2[start_idx:stop_idx]-start_time
y2 = y2[start_idx:stop_idx]
x3 = x3[start_idx:stop_idx]-start_time
y3 = y3[start_idx:stop_idx]
x4 = x4[start_idx:stop_idx]-start_time
y4 = y4[start_idx:stop_idx]

############
# Plotting
###########

brewer_red = [0.89411765, 0.10196078, 0.10980392]
brewer_blue = [0.21568627, 0.49411765, 0.72156863]
brewer_green = [0.30196078, 0.68627451, 0.29019608]

myred = [0.8392156862745098, 0.1568627450980392, 0.1568627450980392]

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

'''
bmap = brewer2mpl.get_map('Spectral', 'diverging', 8)
colors = bmap.mpl_colors
colors = colors[::-1] # Reverse order
'''

'''
colors = [(0.8392156862745098, 0.1568627450980392, 0.1568627450980392), # Red
           config.UIBK_blue,
           config.brighten(config.UIBK_blue, 0.25),
           config.brighten(config.UIBK_blue, 0.5),
           config.brighten(config.UIBK_blue, 0.75)]
'''

colors = [ config.UIBK_orange, 
           config.UIBK_blue,
           config.brighten(config.UIBK_blue, 0.25),
           config.brighten(config.UIBK_blue, 0.5),
           config.brighten(config.UIBK_blue, 0.75)]
           
           

text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0

size_factor = 0.75
figure_width = size_factor*text_width
figure_height = (figure_width / golden_ratio)
#figure_height = (text_width / golden_ratio) # height is golden ratio to page width

#figure_height = 1.3 * figure_width
figure_size = [figure_width, figure_height]

config.load_config_small()
    
fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figure_size, dpi=100)
ax = axes



ax.plot(x, y, linewidth=0.75, color=colors[0], linestyle="-", zorder=1, label='Matrix Mean',
        marker='o', markeredgewidth=0.3, markersize=1.75, markeredgecolor=colors[0], markerfacecolor=[1.0, 1.0, 1.0], alpha=1.0)
        
ax.plot(x_s, y_s, linewidth=0.75, color=brewer_red, linestyle="-", zorder=1, label='Filtered Mean',
        marker='o', markeredgewidth=0.3, markersize=1.75, markeredgecolor=brewer_red, markerfacecolor=[1.0, 1.0, 1.0], alpha=1.0)

ax.plot(x1, y1, linewidth=0.75, color=colors[1], linestyle="-", zorder=1, label='Taxel (4,2)',
        marker='D', markeredgewidth=0.3, markersize=1.75, markeredgecolor=colors[1], markerfacecolor=[1.0, 1.0, 1.0], alpha=1.0)

ax.plot(x2, y2, linewidth=0.75, color=colors[2], linestyle="-", zorder=1, label='Taxel (5,8)',
        marker='p', markeredgewidth=0.3, markersize=1.75, markeredgecolor=colors[2], markerfacecolor=[1.0, 1.0, 1.0], alpha=1.0)

ax.plot(x3, y3, linewidth=0.75, color=colors[3], linestyle="-", zorder=1, label='Taxel (4,6)',
        marker='^', markeredgewidth=0.3, markersize=1.75, markeredgecolor=colors[3], markerfacecolor=[1.0, 1.0, 1.0], alpha=1.0)

ax.plot(x4, y4, linewidth=0.75, color=colors[4], linestyle="-", zorder=1, label='Taxel (5,6)',
        marker='s', markeredgewidth=0.3, markersize=1.75, markeredgecolor=colors[4], markerfacecolor=[1.0, 1.0, 1.0], alpha=1.0)


ax.set_xlim([0, 3])
#ax.set_ylim([0, 850])
ax.set_ylim([0, 3200])
#ax.set_ylim([0, 1.1*ys2.max()])


# Legend
ax.legend(loc = 'upper left', fancybox=True, shadow=False, framealpha=0.75)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Raw Sensor Value", rotation=90)

fig.tight_layout()
#plt.show() 

plotname = "calibration_test"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
