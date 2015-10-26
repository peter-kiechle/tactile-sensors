#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys

# Load configuration file (before pyplot)
config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)
import configuration as config

import numpy as np
import matplotlib.pyplot as plt


# Custom libraries
print("CWD: " + os.getcwd() )
lib_path = os.path.abspath('../../lib')
sys.path.append(lib_path)
import framemanager_python
import module_image_moments as IM
import module_normalized_cross_correlation as NCC


# Reloading not needed in final version
reload(IM)
reload(NCC)
reload(framemanager_python)

def loadFrame(frameManager, frameID, matrixID):
    tsframe = np.copy( frameManager.get_tsframe(frameID, matrixID) );
    # Normalize frame
    tsframe_uint8 = np.uint8(tsframe / (tsframe.max()/255.0)) # scale to [0..255] and convert to uint8
    return tsframe_uint8


def plot_principal_axes_angle(c_x, c_y, major_axis_width, minor_axis_width, angle, color, ax):
   """Plot principal axes using specified angle and length
      Taken from Joe Kington: http://stackoverflow.com/questions/9005659/compute-eigenvectors-of-image-in-python
   """
   def plot_bar(r, c_x, y_bar, angle, ax, color):
        dx = r * np.cos(np.radians(angle))
        dy = r * np.sin(np.radians(angle))
        ax.plot([c_x - dx, c_x, c_x + dx], 
                [c_y - dy, c_y, c_y + dy], '-', color=color, linewidth=2.0)
        """
        ax.annotate("",
                    xy=(c_x-dx, c_y-dy), xycoords='data',
                    xytext=(c_x+dx, c_y+dy), textcoords='data',
                    arrowprops=dict(arrowstyle="|-|", color = color, linewidth=3.0)
                    )        
        """
   plot_bar(minor_axis_width, c_x, c_y, angle+90.0, ax, color=color) # Minor axis
   plot_bar(major_axis_width, c_x, c_y, angle, ax, color=color) # Major axis
   ax.axis('image') # Attach axis to picture dimensions


def plot_principal_axes_cov(x_bar, y_bar, cov, color, ax):
    """Plot principal axes of one stddevs using specified covariance matrix
       Taken from Joe Kington: http://stackoverflow.com/questions/9005659/compute-eigenvectors-of-image-in-python
    """
    def make_lines(eigvals, eigvecs, mean, i):
        """Make lines a length of 1 stddev."""
        std = np.sqrt(eigvals[i])
        vec = 1 * std * eigvecs[:,i] / np.hypot(*eigvecs[:,i])
        x, y = np.vstack((mean-vec, mean, mean+vec)).T
        return x, y
        
    mean = np.array([x_bar, y_bar])
    eigvals, eigvecs = np.linalg.eigh(cov)
    #print eigvals
    artists_minor, = ax.plot(*make_lines(eigvals, eigvecs, mean, 0), color=color, linewidth=2.0)
    artists_major, = ax.plot(*make_lines(eigvals, eigvecs, mean, -1), color=color, linewidth=2.0)
    #ax.axis('image')
    return artists_minor, artists_major
      




##########################
# Load pressure profile
##########################
profileName = os.path.abspath("slip_and_rotation_teapot_handle.dsa")
frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileName);
numFrames = frameManager.get_tsframe_count();

# Relative timestamps in seconds
timestamps = frameManager.get_tsframe_timestamp_list()
timestamps = (timestamps-timestamps[0]) / 1000.0 # Relative timestamps in seconds

matrixID = 1
startID = 13 # 31
stopID = 93


thresh_active_cells_translation = 1; # Minimum amount of active cells for slip vector
thresh_active_cells_rotation = 5; # Minimum amount of active cells for slip angle
thresh_eccentricity = 0.6; # Principal axis lengths (disc or square: 0.0, elongated rectangle: ->1.0)
thresh_compactness = 0.9; # How much the object resembles a disc (perfect circle 1.0)


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
r_slipvector = []
r_slipangle = []
r_centroid_x = []
r_centroid_y = []
r_angle = []
r_Cov = []
r_lambda1 = []
r_lambda2 = []
r_std_dev_x = []
r_std_dev_y = []
r_skew_x = []
r_skew_y = []
r_compactness1 = []
r_compactness2 = []
r_eccentricity1 = []
r_eccentricity2 = []

for frameID in xrange(startID, stopID):
    # Get current frame
    frame1 = loadFrame(frameManager, frameID, matrixID)
    active_cells1 = frameManager.get_num_active_cells_matrix(frameID, matrixID)

    # Compute slip vector
    if (active_cells0 > thresh_active_cells_translation and active_cells1 > thresh_active_cells_translation):
        slipvector = NCC.normalized_cross_correlation(frame0, frame1)
        r_slipvector.append(slipvector)
        frame0 = frame1
        active_cells0 = active_cells1
    else:
        r_slipvector.append(np.zeros(2))
        
    # Compute shape features and orientation
    if active_cells1 > thresh_active_cells_translation:
        (centroid_x, centroid_y, angle, Cov, lambda1, lambda2, 
         std_dev_x, std_dev_y, skew_x, skew_y,
         compactness1, compactness2, eccentricity1, eccentricity2) = IM.compute_orientation_and_shape_features(frame1)
        
        r_centroid_x.append(centroid_x)
        r_centroid_y.append(centroid_y)
        r_angle.append(angle)
        r_Cov.append(Cov)
        r_lambda1.append(lambda1)
        r_lambda2.append(lambda2)
        r_std_dev_x.append(std_dev_x)
        r_std_dev_y.append(std_dev_y)
        r_skew_x.append(skew_x)
        r_skew_y.append(skew_y)
        r_compactness1.append(compactness1)
        r_compactness2.append(compactness2)
        r_eccentricity1.append(eccentricity1)
        r_eccentricity2.append(eccentricity2)

    # Compute slip angle
    if active_cells1 > thresh_active_cells_rotation:
        # Track slip angle
        if IM.valid_frame(compactness2, eccentricity2, thresh_compactness, thresh_eccentricity):   
            current_angle, slip_angle, slip_angle_reference, n = IM.track_angle(reference_angle, previous_angle, angle, n)
            previous_angle = current_angle
            r_slipangle.append(slip_angle)
    else:
        r_slipangle.append(0.0)



r_slipvector = np.array(r_slipvector)
r_slipangle = np.array(r_slipangle)
r_centroid_x = np.array(r_centroid_x)
r_centroid_y = np.array(r_centroid_y)
r_angle = np.array(r_angle)
r_Cov = np.array(r_Cov)
r_lambda1 = np.array(r_lambda1)
r_lambda2 = np.array(r_lambda2)
r_std_dev_x = np.array(r_std_dev_x)
r_std_dev_y = np.array(r_std_dev_y)
r_skew_x = np.array(r_skew_x)
r_skew_y = np.array(r_skew_y)
r_compactness1 = np.array(r_compactness1)
r_compactness2 = np.array(r_compactness2)
r_eccentricity1 = np.array(r_eccentricity1)
r_eccentricity2 = np.array(r_eccentricity2)



cumulative_slipvector = np.cumsum(r_slipvector, axis=0)
cumulative_slipangle = np.cumsum(r_slipangle)



#########
# Report
#########

frameIDS = np.array([13, 22, 41, 71, 86])
timestamps0 = timestamps[startID]

names = ("Centroid x", "Centroid y", "Standard deviation x", "Standard deviation y", "Skewness x", "Skewness y",
         "Angle", "Major axis", "Minor axis", 
         "Compactness (moments)", "Compactness (contour)", "Eccentricity", "Eccentricity (squared)")

for i in frameIDS:

    # Relative timestamps in seconds
    timestamp = timestamps[i] - timestamps[startID]
   
    j = i-startID

    # Regrouping
    features = (r_centroid_x[j], r_centroid_y[j], r_angle[j], r_Cov[j], r_lambda1[j], r_lambda2[j],
                r_std_dev_x[j], r_std_dev_y[j], r_skew_x[j], r_skew_y[j],
                r_compactness1[j], r_compactness2[j], r_eccentricity1[j], r_eccentricity2[j])

    IM.report_shape_features("Frame " + str(i), features)
    print "Time: " + str(timestamp)
    print "Translation: [" + str(cumulative_slipvector[j][1]) + ", " + str(cumulative_slipvector[j][0]) + "]"
    print "Rotation: [" + str(cumulative_slipangle[j]) + "]"

'''
###########
# Plotting
###########
colormap=plt.get_cmap('gray')
#colormap=plt.get_cmap('YlOrRd_r')
#colormap = plt.get_cmap('afmhot')
colormap.set_under([0.0, 0.0, 0.0])
#colormap.set_under([0.3, 0.3, 0.3])
#colormap.set_under([1.0, 1.0, 1.0])

text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0

size_factor = 0.75
figure_width = size_factor*text_width
figure_height = (figure_width / golden_ratio)
#figure_height = 1.3 * figure_width
figure_size = [figure_width, figure_height]

config.load_config_large()



for i in frameIDS:

    frame = loadFrame(frameManager, i, matrixID)
   
    fig = plt.figure(figsize=figure_size)
    ax = plt.subplot()

    # Workaround inverted y-axis
    ax.invert_yaxis()
    #image = np.flipud(image)

    width = frame.shape[1]
    height = frame.shape[0]
    xs,ys = np.meshgrid(np.arange(0, width+1), np.arange(0, height+1))

    # pcolormesh aligns cells on their edges, while imshow aligns them on their centers.
    ax.pcolormesh(xs-0.5, ys-0.5, frame, cmap=colormap, vmin=0.001, vmax=255.0,
                  shading="faceted", linestyle="dashed", linewidth=1.0, edgecolor=[0.0, 0.0, 0.0])

    ax.set_aspect('equal')
    ax.set_xlim([-0.5, width-0.5])
    ax.set_ylim([height-0.5, -0.5])
    ax.xaxis.tick_top()
    plt.tick_params(axis='both', which='both', left='off', right='off', bottom='off', top='off',  labeltop='on')

    plot_principal_axes_cov(r_centroid_x[i-startID], r_centroid_y[i-startID], r_Cov[i-startID], [1.0, 0.5, 0.0, 1.0], ax)
    #plot_principal_axes_angle(centroid_x0, centroid_y0, lambda10, lambda20, angle0, [1.0, 0.5, 0.0, 1.0], ax)

    fig.tight_layout()
    #fig.show()

    plotname = "slipdetection_snap_shot_" + str(i)
    fig.savefig(plotname+".pdf", pad_inches=0, bbox_inches='tight', dpi=fig.dpi) # pdf
    fig.savefig(plotname+".pgf", pad_inches=0, bbox_inches='tight', dpi=fig.dpi) # pgf
    plt.close()
'''