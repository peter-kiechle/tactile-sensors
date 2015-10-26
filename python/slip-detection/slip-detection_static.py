# -*- coding: utf-8 -*-

##########################################
# Load configuration file (before pyplot)
##########################################

import os, sys
config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)
import configuration as config

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

# Custom libraries
print("CWD: " + os.getcwd() )
lib_path = os.path.abspath('../../lib')
sys.path.append(lib_path)
import framemanager_python
import module_image_moments as IM
import module_normalized_cross_correlation as NCC


# Force reloading of external library (convenient during active development)
reload(IM)
reload(NCC)
reload(framemanager_python)

def loadFrame(frameManager, frameID, matrixID):
    tsframe = np.copy( frameManager.get_tsframe(frameID, matrixID) );
    # Normalize frame
    #tsframe_uint8 = np.uint8(tsframe / (4096.0/255.0)) # scale to [0..255] and convert to uint8
    # tsframe /= 4096.0 # scale to [0..1]
    return tsframe



############
# Settings
############

matrixID = 1
startID = 13 # slip_and_rotation_teapot_handle
stopID = 93 # slip_and_rotation_teapot_handle
#startID = 22 # slip_examples_pen_and_wooden_block_000093-000189.dsa
#stopID = 80 # slip_examples_pen_and_wooden_block_000093-000189.dsa

thresh_active_cells_translation = 1; # Minimum amount of active cells for slip vector
thresh_active_cells_rotation = 5; # Minimum amount of active cells for slip angle
thresh_eccentricity = 0.6; # Principal axis lengths (disc or square: 0.0, elongated rectangle: ->1.0)
thresh_compactness = 0.9; # How much the object resembles a disc (perfect circle 1.0)


########################
# Load pressure profile
########################

profileName = os.path.abspath("slip_and_rotation_teapot_handle.dsa")
#profileName = os.path.abspath("slip_examples_pen_and_wooden_block_000093-000189.dsa")

frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileName);
numFrames = frameManager.get_tsframe_count();

# Relative timestamps in seconds
timestamps = frameManager.get_tsframe_timestamp_list()[startID:stopID]
timestamps = (timestamps-timestamps[0]) / 1000.0

# Get initial frame (Assumtion: valid frame)
frame0 = loadFrame(frameManager, startID, matrixID)
active_cells0 = frameManager.get_num_active_cells_matrix(startID, matrixID)

# Compute orientation of initial frame
(centroid_x, centroid_y, angle, Cov, lambda1, lambda2, 
 std_dev_x, std_dev_y, skew_x, skew_y,
 compactness1, compactness2, eccentricity1, eccentricity2) = IM.compute_orientation_and_shape_features(frame0)


reference_angle = angle # [0, 180)
previous_angle = angle # [0, 360)
slip_angle = 0 # (-∞, ∞)
n = 0 # Rotation carry



# Records
slipvectors = np.zeros([1,2])
slipvectors_ncc_1 = np.zeros([1,2])
slipvectors_ncc_2 = np.zeros([1,2])
slipvectors_pc = np.zeros([1,2])
slipangles = np.zeros([1,1])

slipvectors_delta = np.zeros([1,2])
slipvectors_ncc_1_delta = np.zeros([1,2])
slipvectors_ncc_2_delta = np.zeros([1,2])
slipvectors_pc_delta = np.zeros([1,2])
slipangles_delta = np.zeros([1,1])

centroids = np.array([centroid_x, centroid_y])


for frameID in xrange(startID+1, stopID):
    # Get current frame
    frame1 = loadFrame(frameManager, frameID, matrixID)
    active_cells1 = frameManager.get_num_active_cells_matrix(frameID, matrixID)

    # Compute slip vector
    if (active_cells0 > thresh_active_cells_translation and active_cells1 > thresh_active_cells_translation):
        slipvector = NCC.normalized_cross_correlation(frame0, frame1)
        slipvectors_delta = np.vstack((slipvectors_delta, slipvector))
        slipvectors = np.vstack((slipvectors, slipvectors[-1]+slipvector))
        
        slipvector_ncc_1 = NCC.normalized_cross_correlation2(frame0, frame1)
        slipvectors_ncc_1_delta = np.vstack((slipvectors_ncc_1_delta, slipvector_ncc_1))
        slipvectors_ncc_1 = np.vstack((slipvectors_ncc_1, slipvectors_ncc_1[-1]+slipvector_ncc_1))
        
        slipvector_ncc_2 = NCC.normalized_cross_correlation3(frame0, frame1)
        slipvectors_ncc_2_delta = np.vstack((slipvectors_ncc_2_delta, slipvector_ncc_2))
        slipvectors_ncc_2 = np.vstack((slipvectors_ncc_2, slipvectors_ncc_2[-1]+slipvector_ncc_2))

        slipvector_pc = NCC.normalized_cross_correlation4(frame0, frame1)
        slipvectors_pc_delta = np.vstack((slipvectors_pc_delta, slipvector_pc))
        slipvectors_pc = np.vstack((slipvectors_pc, slipvectors_pc[-1]+slipvector_pc))
     
        frame0 = frame1
        active_cells0 = active_cells1
    else:
        slipvectors_delta = np.vstack((slipvectors_delta, np.zeros(2)))
        slipvectors = np.vstack((slipvectors, slipvectors[-1]))
        
        slipvectors_ncc_1_delta = np.vstack((slipvectors_ncc_1_delta, np.zeros(2)))
        slipvectors_ncc_1 = np.vstack((slipvectors_ncc_1, slipvectors_ncc_1[-1]))
        
        slipvectors_ncc_2_delta = np.vstack((slipvectors_ncc_2_delta, np.zeros(2)))
        slipvectors_ncc_2 = np.vstack((slipvectors_ncc_2, slipvectors_ncc_2[-1]))
        
        slipvectors_pc_delta = np.vstack((slipvectors_pc_delta, np.zeros(2)))
        slipvectors_pc = np.vstack((slipvectors_pc, slipvectors_pc[-1]))


    # Compute shape features and orientation
    if active_cells1 > thresh_active_cells_translation:
        (centroid_x, centroid_y, angle, Cov, lambda1, lambda2, 
         std_dev_x, std_dev_y, skew_x, skew_y,
         compactness1, compactness2, eccentricity1, eccentricity2) = IM.compute_orientation_and_shape_features(frame1)
        
        # Record center of mass movement for comparison with normalized cross correlation
        centroids = np.vstack((centroids, [centroid_x, centroid_y]))

    # Compute slip angle
    if active_cells1 > thresh_active_cells_rotation:
        # Track slip angle
        if IM.valid_frame(compactness2, eccentricity2, thresh_compactness, thresh_eccentricity):   
            current_angle, slip_angle, slip_angle_reference, n = IM.track_angle(reference_angle, previous_angle, angle, n)
            previous_angle = current_angle
            slipangles_delta = np.vstack((slipangles_delta, slip_angle))
            slipangles = np.vstack((slipangles, slipangles[-1] + slip_angle))
    else:
        slipangles_delta = np.vstack((slipangles_delta, np.zeros(1)))
        slipangles = np.vstack((slipangles, slipangles[-1]))



slipvectors *= 3.4 # convert from cells to millimeter
slipvectors_ncc_1 *= 3.4
slipvectors_ncc_2 *= 3.4
slipvectors_pc *= 3.4

'''
# Center of mass based slip
centroids_diff = np.diff(centroids, axis=0)
centroids_cumsum = np.cumsum(centroids_diff, axis=0)
slipvector_ncc_2s = np.vstack(([0.0, 0.0], centroids_cumsum))
slipvector_diff = np.abs(slipvectors) - np.abs(slipvector_ncc_2s)
'''


############
# Plotting
############ 

#------------------------------------------------------------------------
# All in one
#------------------------------------------------------------------------   
     
text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0

size_factor = 0.45
figure_width = size_factor*text_width
#figure_height = (figure_width / golden_ratio)
figure_height = 2.2 * figure_width
figure_size = [figure_width, figure_height]

config.load_config_medium()

fig = plt.figure(figsize=figure_size)

ax1 = plt.subplot(3,1,1)
x = timestamps[0:stopID-startID]
#y = slipvectors_delta[:,0]
ax1.plot(x, slipvectors[:,0], label="Alcazar", ls="-", lw=1.5, color=config.UIBK_blue, alpha=0.75)
ax1.plot(x, slipvectors_ncc_1[:,0], label="NCC 1", ls="-", lw=1.5, color=config.UIBK_orange, alpha=1.0)
ax1.plot(x, slipvectors_ncc_2[:,0], label="NCC 2", ls="-", lw=1.5, dashes=[3,1], color=config.UIBK_orange, alpha=1.0)
ax1.plot(x, slipvectors_pc[:,0], label="PC", ls="-", lw=1.5, color=[0.0, 0.0, 0.0], alpha=1.0)
ax1.set_ylabel(r"$\Delta x$ [mm]", rotation=90)
ax1.yaxis.set_major_locator(MaxNLocator(integer=True)) # Restriction to integer
ax1.legend(fontsize=8, loc='upper left', fancybox=True, shadow=False, framealpha=1.0)
#ax1.grid('on')

#ax2 = plt.subplot(3,1,2, sharex=ax1, sharey=ax1)
ax2 = plt.subplot(3,1,2)
x = timestamps[0:stopID-startID]
#y = slipvectors_delta[:,1]
ax2.plot(x, slipvectors[:,1], label="Alcazar", ls="-", lw=1.5, color=config.UIBK_blue, alpha=0.75)
ax2.plot(x, slipvectors_ncc_1[:,1], label="NCC 1", ls="-", lw=1.5, color=config.UIBK_orange, alpha=1.0)
ax2.plot(x, slipvectors_ncc_2[:,1], label="NCC 2", ls="-", dashes=[3,1],  lw=1.5, color=config.UIBK_orange, alpha=1.0)
ax2.plot(x, slipvectors_pc[:,1], label="PC", ls="-", lw=1.5, color=[0.0, 0.0, 0.0], alpha=1.0)
ax2.set_ylabel(r"$\Delta y$ [mm]", rotation=90)
ax2.yaxis.set_major_locator(MaxNLocator(integer=True)) # Restriction to integer
ax2.legend(fontsize=8, loc='upper left', fancybox=True, shadow=False, framealpha=1.0)
#ax2.grid('on')
#ax2.set_ylim([0, 22])


ax3 = plt.subplot(3,1,3, sharex=ax1)
x = timestamps[0:stopID-startID]
#y = slipangles_delta
y = slipangles
ax3.plot(x, y, label=r"$\theta$", lw=1.5, color=config.UIBK_orange, alpha=1.0)
ax3.set_ylabel("Rotation", rotation=90)
ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter(r"%.1f$^\circ$"))
#ax3.grid('on')

x_poi = x[74]
y_poi = y[74]
ax3.annotate(r"Invalid shape", size=8,
            xy=(x_poi, y_poi), xycoords='data', 
            xytext=(0, 43), textcoords='offset points', ha="right", va="center",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", fc="black"),
            )



ax3.set_xlabel('Time [s]')

#plt.subplots_adjust(top=0.98, bottom=0.08, left=0.08, right=0.98, wspace=0.0, hspace=0.2)
fig.tight_layout()

plotname = "all_in_one_ncc"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf


#------------------------------------------------------------------------
# Slip vector
#------------------------------------------------------------------------
config.load_config_medium()

# Slip vector
text_width = 6.30045 # LaTeX text width in inches
figure_width = 0.45 * text_width
figure_height = figure_width
figure_size = [figure_width, figure_height]

fig, ax = plt.subplots(figsize=figure_size)
ax.plot(slipvectors[:,0], slipvectors[:,1], label="Alcazar", linewidth=1.5, color=config.UIBK_blue, alpha=0.75)
#ax.plot(slipvectors_ncc[:,0], slipvectors_ncc[:,1], label="NCC", ls="-", dashes=[2,1], lw=1.0, color=config.UIBK_blue, alpha=1.0)
ax.plot(slipvectors_ncc_1[:,0], slipvectors_ncc_1[:,1], label="NCC 1", ls="-", lw=1.5, color=config.UIBK_orange, alpha=1.0)
ax.plot(slipvectors_ncc_2[:,0], slipvectors_ncc_2[:,1], label="NCC 2", ls="-", dashes=[3,1], lw=1.5, color=config.UIBK_orange, alpha=1.0)
ax.plot(slipvectors_pc[:,0], slipvectors_pc[:,1], label="PC", ls="-", lw=1.5, color=[0.0, 0.0, 0.0], alpha=1.0, zorder=0)
ax.axis('equal')
ax.set_xlabel(r"$\Delta x$ [mm]")
ax.set_ylabel(r"$\Delta y$ [mm]", rotation=90)
ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # Restriction to integer
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

ax.legend(fontsize=8, loc='lower right', fancybox=True, shadow=False, framealpha=1.0)

#ax.yaxis.labelpad = 10
#ax.grid('on')
#fig.tight_layout()
#ax.set_title("Slip trajectory")
#plt.show()
plt.subplots_adjust(top=0.85, left = 0.15, bottom=0.15, right = 0.85)  # Legend on top
plotname = "slip_trajectory_ncc"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf




#------------------------------------------------------------------------
# Rotation
#------------------------------------------------------------------------

# Thanks to Joe Kington
# http://stackoverflow.com/questions/20222436/python-matplotlib-how-to-insert-more-space-between-the-axis-and-the-tick-labe
def realign_polar_xticks(ax):
    for theta, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        theta = theta * ax.get_theta_direction() + ax.get_theta_offset()
        theta = np.pi/2 - theta
        y, x = np.cos(theta), np.sin(theta)
        if x >= 0.1:
            label.set_horizontalalignment('left')
        if x <= -0.1:
            label.set_horizontalalignment('right')
        if y >= 0.5:
            label.set_verticalalignment('bottom')
        if y <= -0.5:
            label.set_verticalalignment('top')

r = np.sqrt(slipvectors[:,0]**2 + slipvectors[:,1]**2)
r_ncc_1 = np.sqrt(slipvectors_ncc_1[:,0]**2 + slipvectors_ncc_1[:,1]**2)
r_ncc_2 = np.sqrt(slipvectors_ncc_2[:,0]**2 + slipvectors_ncc_2[:,1]**2)
r_pc = np.sqrt(slipvectors_pc[:,0]**2 + slipvectors_pc[:,1]**2)
theta = np.deg2rad(slipangles)
d = r.max() / 3.4 # max distance

fig, ax = plt.subplots(figsize=figure_size)
ax = plt.subplot(111, polar=True)
ax.plot(theta, r, label="Alcazar", lw=1.5, color=config.UIBK_blue, alpha=0.75)
ax.plot(theta, r_ncc_1, label="NCC 1", ls="-", lw=1.5, color=config.UIBK_orange, alpha=1.0)
ax.plot(theta, r_ncc_2, label="NCC 2", ls="-", dashes=[3,1],  lw=1.5, color=config.UIBK_orange, alpha=1.0)
ax.plot(theta, r_pc, label="PC", ls="-", lw=1.5, color=[0.0, 0.0, 0.0], alpha=1.0)
ax.set_rmax(d)

# tick labels (Workaround for pgf degree symbol)
xtick_labels=[r"0$^\circ$", r"45$^\circ$",
            r"90$^\circ$", r"135$^\circ$",
            r"180$^\circ$", r"225$^\circ$",
            r"270$^\circ$", r"315$^\circ$"]
    
ax.set_xticklabels(xtick_labels)

# tick locations
thetaticks = np.arange(0,360,45)
ax.set_thetagrids(thetaticks, frac=1.1) # additional distance
realign_polar_xticks(ax)

#ax.grid(True)
#ax.set_title("Rotation")
fig.tight_layout()
plt.rgrids(np.arange(1.0, d+1, 1), angle=90);
ax.set_yticklabels( [("%.1f mm" % i) for i in 3.4*np.arange(1,d)], fontsize=6)

ax.legend(fontsize=8, bbox_to_anchor=[0.25, 0.5], loc='center', fancybox=True, shadow=False, framealpha=1.0)


plotname = "rotation_trajectory_ncc"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()


