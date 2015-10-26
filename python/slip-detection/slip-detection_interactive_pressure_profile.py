#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Load configuration file (before pyplot)
#execfile('../matplotlib/configuration.py')
import os, sys
config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)
import configuration as config

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec

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
    tsframe_uint8 = np.uint8(tsframe / (4096.0/255.0)) # scale to [0..255] and convert to uint8
    return tsframe_uint8


def plot_principal_axes_angle(c_x, c_y, major_axis_width, minor_axis_width, angle, color, ax):
   """Plot principal axes using specified angle and length
      Taken from Joe Kington: http://stackoverflow.com/questions/9005659/compute-eigenvectors-of-image-in-python
   """
   def plot_bar(r, c_x, y_bar, angle, ax, color):
        dx = r * np.cos(np.radians(angle))
        dy = r * np.sin(np.radians(angle))
        ax.plot([c_x - dx, c_x, c_x + dx], 
                [c_y - dy, c_y, c_y + dy], '-', color=color, linewidth=3.0)
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
    artists_minor, = ax.plot(*make_lines(eigvals, eigvecs, mean, 0), color=color, linewidth=3.0)
    artists_major, = ax.plot(*make_lines(eigvals, eigvecs, mean, -1), color=color, linewidth=3.0)
    #ax.axis('image')
    return artists_minor, artists_major
      

def plot_frame(ax, frame, centroid_x, centroid_y, Cov):
    
    ax.cla()    
    
    ax.imshow(frame, cmap=cmap, vmin=0.001, vmax=255.0, interpolation='nearest')
    # Save current axis-limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
   
    l1, l2 = plot_principal_axes_cov(centroid_x, centroid_y, Cov, config.UIBK_orange, ax)
    
    # Create new axes on the right and on the top of the current axis
    divider = make_axes_locatable(ax)
    ax_x = divider.append_axes("top", size=1.0, pad=0.2, sharex=ax)
    ax_y = divider.append_axes("right", size=1.0, pad=0.2, sharey=ax)    
    
    # Axis projection
    xp = np.sum(frame, axis=0)
    xs = np.arange(0, frame.shape[1])  
    yp = np.sum(frame, axis=1)
    ys = np.arange(0, frame.shape[0])  

    # Histogram style
    ax_x.bar(xs, xp, align="center", color=[0.8, 0.8, 0.8], ec=[0.0, 0.0, 0.0], lw=1.0, width=1.0 )
    ax_y.barh(ys, yp, align="center", color=[0.8, 0.8, 0.8], ec=[0.0, 0.0, 0.0], lw=1.0, height=1.0 )
    
    # Filled Curve style
    # Define corner points for filled polygon (fill_between() is buggy)
    #xs2 = np.concatenate(([0], xs, [xs[-1]]))
    #xp2 = np.concatenate(([0], xp, [0]))
    #ys2 = np.concatenate(([0], ys, [ys[-1]]))
    #yp2 = np.concatenate(([0], yp, [0]))
    #ax_x.fill(xs2,xp2, color=[0.8, 0.8, 0.8])
    #ax_x.plot(xs2,xp2, lw=2.0, color=[0.0, 0.0, 0.0])
    #ax_y.fill(yp2, ys2, color=[0.8, 0.8, 0.8])
    #ax_y.plot(yp2, ys2, lw=2.0, color=[0.0, 0.0, 0.0])
    
    # Restore axis limits
    ax_x.set_xlim(xlim)
    ax_y.set_ylim(ylim)
        
    max_intensity = np.max(np.concatenate((xp, yp)))
    ax_y.set_xlim([0, 1.1*max_intensity])
    ax_x.set_ylim([0, 1.1*max_intensity])
        
    # Tick label locations
    ax.set_xticks(xs)
    ax.set_yticks(ys)
    ax_x.yaxis.set_major_locator(MaxNLocator(5))
    ax_y.xaxis.set_major_locator(MaxNLocator(5))

    # Remove duplicated tick labels
    plt.setp(ax_x.get_xticklabels() + ax_x.get_yticklabels() + ax_y.get_xticklabels() + ax_y.get_yticklabels(), visible=False)

    ax.set_xticks(xs)
    ax.set_yticks(ys)



def update_frame(val):
    global n  
    global reference_angle    
    global previous_angle    
    
    reference_frameID = int(np.around(slider_reference_frame.val))
    current_frameID = int(np.around(slider_current_frame.val))

    frame0 = loadFrame(frameManager, reference_frameID, matrixID)
    frame1 = loadFrame(frameManager, current_frameID, matrixID)
    
    # Compute translation between reference frames
    slipvector = NCC.normalized_cross_correlation(frame0, frame1)
    
    
    # Compute orientation of reference frame
    (centroid_x0, centroid_y0, angle0, Cov0, lambda10, lambda20, 
     std_dev_x0, std_dev_y0, skew_x0, skew_y0,
     compactness10, compactness20, eccentricity10, eccentricity20) = IM.compute_orientation_and_shape_features(frame0)

    # Compute orientation of current frame
    features1 = IM.compute_orientation_and_shape_features(frame1)
    (centroid_x1, centroid_y1, angle1, Cov1, lambda11, lambda21, 
     std_dev_x1, std_dev_y1, skew_x1, skew_y1,
     compactness11, compactness21, eccentricity11, eccentricity21) = features1
    
    IM.report_shape_features("Frame %d" %current_frameID , features1)
    
    # Reset tracking if reference angle changed
    if abs(reference_angle-angle0) > 0.001: 
        reference_angle = angle0 # [0, 180)
        previous_angle = angle0 # [0, 360)
        slip_angle = 0 # (-∞, ∞)
        n = 0 # rotation carry
     
    # Track rotation 
    current_angle, slip_angle, slip_angle_reference, n = IM.track_angle(reference_angle, previous_angle, angle1, n)
    previous_angle = current_angle

    ###########
    # Plotting
    ###########
   
    # Reference frame
    plot_frame(axes[1], frame0, centroid_x0, centroid_y0, Cov0)
    
    # Current frame
    plot_frame(axes[2], frame1, centroid_x1, centroid_y1, Cov1)

    # Textbox
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=1.0)
    ax = axes[0]    
    ax.cla()
    ax.axis('off')
    ax.text(0.5, 0.5, text_template%(current_angle, slip_angle, slipvector[0], slipvector[1]), 
                 transform=ax.transAxes, ha="center", va="center", size=14, bbox=bbox_props)
    plt.draw()


  


##########################
# Load pressure profile
##########################
profileName = os.path.abspath("slip_and_rotation_teapot_handle.dsa")
frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileName);
numFrames = frameManager.get_tsframe_count();

reference_frameID = 13
current_frameID = 13
matrixID = 1

frame0 = loadFrame(frameManager, reference_frameID, matrixID)

# Compute initial orientation
features = IM.compute_orientation_and_shape_features(frame0)
(centroid_x, centroid_y, angle, Cov, lambda1, lambda2, 
 std_dev_x, std_dev_y, skew_x, skew_y,
 compactness1, compactness2, eccentricity1, eccentricity2) = features

IM.report_shape_features("Initial frame", features)

reference_angle = angle # [0, 180)
previous_angle = angle # [0, 360)
slip_angle = 0 # (-∞, ∞)
n = 0 # rotation carry



###########
# Plotting
###########

#mpl.rcParams['text.usetex'] = True  # True, False
#text_template_angle = r'$\mathbf{ \phi: %.1f \quad \Delta \phi = %.1f} $'
#text_template_shift = r'$\mathbf{ \Delta x: %.1f, \quad \Delta y: %.1f} $'
#text_template_angle = 'phi: %.1f   dphi = %.1f'
#text_template_shift = 'dx: %.1f,   dy: %.1f'
text_template = r"$\theta$: %03.1f    $\Delta\theta$ = %03.1f"
text_template += "\n"
text_template += r"$\Delta$x: %.1f    $\Delta$y: %.1f"

cmap=plt.get_cmap('gray')
cmap.set_under([0.0, 0.0, 0.0])




fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, squeeze=True, dpi=100)


gs = gridspec.GridSpec(2, 2, height_ratios=[1,10])
ax1 = plt.subplot(gs[0, :]) # top
ax2 = plt.subplot(gs[1, 0]) # left
ax3 = plt.subplot(gs[1, 1]) # right
axes = np.array([ax1, ax2, ax3])


fig = plt.subplots_adjust(top=0.95, left = 0.07, bottom=0.2, right = 1.0)

ax_slider_reference = plt.axes([0.25, 0.10, 0.6, 0.03]) # left, bottom, width, height
ax_slider_current = plt.axes([0.25, 0.05, 0.6, 0.03]) # left, bottom, width, height

slider_reference_frame = Slider(ax_slider_reference, 'Reference Frame', 0, numFrames-1, valfmt='%0.0f', valinit=reference_frameID)
slider_current_frame = Slider(ax_slider_current, 'Current Frame', 0, numFrames-1, valfmt='%0.0f', valinit=current_frameID)

slider_reference_frame.on_changed(update_frame)
slider_current_frame.on_changed(update_frame)

update_frame(42)

plt.show()
