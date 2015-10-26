# -*- coding: utf-8 -*-

#!/usr/bin/env python

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
import colorsys

import pygtk
pygtk.require('2.0')
import gtk

# Custom library
print("CWD: " + os.getcwd() )
lib_path = os.path.abspath('../../lib')
sys.path.append(lib_path)
import framemanager_python


##############
# Color stuff
##############

UIBK_blue = [0.0, 0.1765, 0.4392]
UIBK_orange = [1.0, 0.5, 0.0]

def change_brightness(colors, factor):
    brighter_colors = []
    for color in colors:
        color = colorsys.rgb_to_hls(color[0], color[1], color[2])
        # workaround immutable tuples        
        color = list(color)
        color[1] = np.min([1.0, factor*color[1]])
        color = tuple(color)
        brighter_colors.append( colorsys.hls_to_rgb(color[0], color[1], color[2]) )
    return brighter_colors

# Define YELLOW_RED colormap:
# For each RGB channel: each row consists of (x, y0, y1) where the x must increase from 0 to 1
#row i:    x  y0  y1
#               /
#              /
#row i+1:  x  y0  y1
cdict = {'red':   ((0.0, 0.9, 0.9),  # Red channel remains constant
                   (1.0, 0.9, 0.9)), 
         'green': ((0.0, 0.9, 0.9),  # Green fades out
                   (1.0, 0.0, 0.0)),
         'blue':  ((0.0, 0.0, 0.0),  # Blue is turned off
                   (1.0, 0.0, 0.0))}
plt.register_cmap(name='YELLOW_RED', data=cdict)

colormap = plt.get_cmap('YELLOW_RED')
#colormap = plt.get_cmap('gnuplot')
#colormap = plt.get_cmap('YlOrRd')
#colormap = plt.get_cmap('autumn')
#colormap = plt.get_cmap('afmhot')
#colormap = plt.get_cmap('gist_heat')
#colormap = plt.get_cmap('gray')

# Color of inactive cells
#colormap.set_under([0.0, 0.0, 0.0])
colormap.set_under([0.2, 0.2, 0.2]) 

# Categorical colors
colors = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), # Blue
          (1.0, 0.4980392156862745, 0.054901960784313725), # Orange
          (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), # Green
          (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), # Red
          (0.5803921568627451, 0.403921568627451, 0.7411764705882353), # Purple
          (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), # Brown
          (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), # Greenish/yellow
          (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)] # Aquamarine

brighter_colors = change_brightness(colors, 1.5)





##############
# Filechooser
##############
def pick_file():

    filename = None   
    
    # Check for new pygtk: this is new class in PyGtk 2.4
    if gtk.pygtk_version < (2,3,90):
        print "PyGtk 2.3.90 or later required for this example"
        raise SystemExit

    dialog = gtk.FileChooserDialog("Open..",
                               None,
                               gtk.FILE_CHOOSER_ACTION_OPEN,
                               (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                gtk.STOCK_OPEN, gtk.RESPONSE_OK))
                                
    dialog.set_default_response(gtk.RESPONSE_OK)
    dialog.set_current_folder(os.getcwd())

    filter = gtk.FileFilter()
    filter.set_name("DSA Pressure Profiles")
    filter.add_pattern("*.dsa")
    dialog.add_filter(filter)
    
    filter = gtk.FileFilter()
    filter.set_name("All files")
    filter.add_pattern("*")
    dialog.add_filter(filter)
    
    response = dialog.run()
    if response == gtk.RESPONSE_OK:
        filename = dialog.get_filename()
    #elif response == gtk.RESPONSE_CANCEL:
    #    sys.exit()
        
    dialog.destroy()
    return filename


    
    
    
###########################
# Called when slider moves
###########################
def update_frame(val):
    frameID = int(slider_frameID.val)
    for matrixID, name in enumerate(matrix_description):
        frame = frameManager.get_tsframe(frameID, matrixID)
        ax = axismapping[matrixID]
        ax.cla() 
        ax.imshow(frame, cmap=colormap, vmin=0.001, vmax=maxValue, interpolation='nearest')
        ax.text(0.5, 0.5, "%d" % matrixID, va="center", ha="center", color=[1.0, 1.0, 1.0, 0.5], fontsize=32, transform=ax.transAxes)
        # Remove axis labels
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)

    marker = ax_graph.axvline(x=timestamps[frameID], ymin=0.0, ymax = 1.0, lw=2, ls='--', color=[1.0, 1.0, 1.0], alpha=0.5)    
    plt.draw()
    marker.remove()


    


# Load pressure profile
#file_name = "foam_ball.dsa"
if os.environ.get('DSA_PROFILE_NAME'):
    file_name = os.environ['DSA_PROFILE_NAME']
else:
    file_name = pick_file()
    
if file_name == None:
    sys.exit()

print "Opening file: ", file_name

profileAbsPath = os.path.abspath(file_name)
profileName = os.path.basename(profileAbsPath)
frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileAbsPath);
numFrames = frameManager.get_tsframe_count();
maxValue = np.max(frameManager.get_max_frame_list())

# Matrix averages
averages_matrix_0 = frameManager.get_average_matrix_list(0)
averages_matrix_1 = frameManager.get_average_matrix_list(1)
averages_matrix_2 = frameManager.get_average_matrix_list(2)
averages_matrix_3 = frameManager.get_average_matrix_list(3)
averages_matrix_4 = frameManager.get_average_matrix_list(4)
averages_matrix_5 = frameManager.get_average_matrix_list(5)

# Time stamps
timestamps = frameManager.get_tsframe_timestamp_list()
timestamps = (timestamps-timestamps[0]) / 1000.0 # Relative timestamps in seconds



############
# Plotting
############

# Grid-coordinates: (y, x)
gs = gridspec.GridSpec(1, 2, wspace=0.25, width_ratios=[1,1])
gs_left = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0])
gs_right = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1])

ax_finger_0_dist = plt.subplot(gs_left[0,0])
ax_finger_0_prox = plt.subplot(gs_left[1,0])
ax_finger_1_dist = plt.subplot(gs_left[0,1])
ax_finger_1_prox = plt.subplot(gs_left[1,1])
ax_finger_2_dist = plt.subplot(gs_left[0,2])
ax_finger_2_prox = plt.subplot(gs_left[1,2])
ax_graph = plt.subplot(gs_right[0])

axismapping = [ax_finger_0_prox, ax_finger_0_dist,
               ax_finger_1_prox, ax_finger_1_dist,
               ax_finger_2_prox, ax_finger_2_dist, ax_graph]

matrix_description = ["Finger 0: Proximal", "Finger 0: Distal",
                      "Finger 1: Proximal", "Finger 1: Distal",
                      "Finger 2: Proximal", "Finger 2: Distal"]

axismapping[0].set_xlabel("Finger 0")
axismapping[2].set_xlabel("Finger 1")
axismapping[4].set_xlabel("Finger 2")
axismapping[0].set_ylabel("Proximal")
axismapping[1].set_ylabel("Distal")
    
# Plot matrix averages
ax_graph.plot(timestamps, averages_matrix_0, lw=1, label="Average Matrix 0", color=colors[0])
ax_graph.plot(timestamps, averages_matrix_1, lw=1, label="Average Matrix 1", color=brighter_colors[0])
ax_graph.plot(timestamps, averages_matrix_2, lw=1, label="Average Matrix 2", color=colors[2])
ax_graph.plot(timestamps, averages_matrix_3, lw=1, label="Average Matrix 3", color=brighter_colors[2])
ax_graph.plot(timestamps, averages_matrix_4, lw=1, label="Average Matrix 4", color=colors[3])
ax_graph.plot(timestamps, averages_matrix_5, lw=1, label="Average Matrix 5", color=brighter_colors[3])
ax_graph.set_axis_bgcolor([0.2, 0.2, 0.2])
#ax_graph.legend()
#ax_graph.legend(loc = 'upper left')
ax_graph.set_xlabel("Time [s]")
ax_graph.set_ylabel("Matrix Average", rotation=90)

plt.subplots_adjust(top=0.90, left = 0.05, bottom=0.15, right = 0.95)
ax_slider = plt.axes([0.25, 0.02, 0.6, 0.03]) # left, bottom, width, height
slider_frameID = Slider(ax_slider, 'Frame ID', 0, numFrames-1, valfmt='%0.0f', valinit=0)
slider_frameID.on_changed(update_frame)
update_frame(0)

plt.suptitle("Profile: "+profileName, fontsize=16)
  
plt.show()
