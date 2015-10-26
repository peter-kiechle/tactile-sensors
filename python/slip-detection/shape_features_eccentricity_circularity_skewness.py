# -*- coding: utf-8 -*-


# Load configuration file (before pyplot)
import os, sys
config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)
import configuration as config

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

# Custom libraries
print("CWD: " + os.getcwd() )
lib_path = os.path.abspath('../../lib')
sys.path.append(lib_path)
import framemanager_python
import module_image_moments as IM

# Reloading not needed in final version
reload(IM)
reload(framemanager_python)

cdict = {'red':   ((0.0, 0.9, 0.9),  # Red channel remains constant
                   (1.0, 0.9, 0.9)), 
         'green': ((0.0, 0.9, 0.9),  # Green fades out
                   (1.0, 0.0, 0.0)),
         'blue':  ((0.0, 0.0, 0.0),  # Blue is turned off
                   (1.0, 0.0, 0.0))}
plt.register_cmap(name='YELLOW_RED', data=cdict)






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
    artists_minor, = ax.plot(*make_lines(eigvals, eigvecs, mean, 0), color=color, linewidth=3.0)
    artists_major, = ax.plot(*make_lines(eigvals, eigvecs, mean, -1), color=color, linewidth=3.0)
    return artists_minor, artists_major


##################
# Experiments
##################
filenames = ( "img_square.png",
              "img_square_or_disc.png", 
              "img_rectangle.png",
              "img_doughnut.png", 
              "img_disc_symmetric.png",
              "img_disc_asymmetric.png",
              #"img_test01.png",
              "img_test02.png",
              "img_test03.png",
              "img_test04.png",
              "img_test05.png",
              "img_test06.png"              
            )

frames = []
for name in filenames:
    frame = cv2.imread(name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    frames.append( (name, frame) ) 

for frame in frames:
    # Compute shape features and orientation
    features = IM.compute_orientation_and_shape_features(frame[1])
    # Report
    IM.report_shape_features(frame[0], features)






########################
# Load pressure profile
########################

profileName = os.path.abspath("slip_and_rotation_teapot_handle.dsa")
frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileName);
frameID = 34
matrixID = 1

frame = np.copy( frameManager.get_tsframe(frameID, matrixID) );
frame = np.uint8(frame / (4096.0/255.0)) # scale to [0..255] and convert to uint8
#frame = cv2.normalize(frame, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.cv.CV_8UC1); # Scale such that highest intensity is 255

# Compute shape features and orientation
features = IM.compute_orientation_and_shape_features(frame)

# Unpacking...
(centroid_x, centroid_y, angle, Cov, lambda1, lambda2, 
 std_dev_x, std_dev_y, skew_x, skew_y,
 compactness1, compactness2, eccentricity1, eccentricity2) = features
 
 
 
###########
# Plotting
###########
UIBK_blue = [0.0, 0.1765, 0.4392]
UIBK_orange = [1.0, 0.5, 0.0]

colormap = plt.get_cmap('gray')
colormap.set_under([0.0, 0.0, 0.0])
colormap = plt.get_cmap('YELLOW_RED')
colormap.set_under([0.2, 0.2, 0.2])


fig_anim, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=(10,10), dpi=100)
ax.imshow(frame, cmap=colormap, vmin=0.001, vmax=np.max(frame), interpolation='nearest')

# Save current axis-limits
xlim = ax.get_xlim()
ylim = ax.get_ylim()

l1, l2 = plot_principal_axes_cov(centroid_x, centroid_y, Cov, UIBK_orange, ax)

# Create new axes on the right and on the top of the current axis
divider = make_axes_locatable(ax)
ax_x = divider.append_axes("top", size=2.0, pad=0.2, sharex=ax)
ax_y = divider.append_axes("right", size=2.0, pad=0.2, sharey=ax)
   
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

plt.show()

