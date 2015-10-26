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


import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma

# Load frame
profileName = os.path.abspath("ghosting_threshold_50.dsa")
frameID = 60

#profileName = os.path.abspath("wooden_block_flat_001968-002012.dsa")
#frameID = 29

frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileName);
numTSFrames = frameManager.get_tsframe_count();

matrixID = 1




# Load single frame
tsframe = np.copy( frameManager.get_tsframe(frameID, matrixID) );

averages = frameManager.get_average_matrix_list(matrixID)

# Load entire profile 
width = tsframe.shape[1]
height = tsframe.shape[0]
tsframes3D = np.empty((height, width, numTSFrames)) # height, width, depth
for i in range(numTSFrames):
    tsframes3D[:,:,i] = np.copy( frameManager.get_tsframe(i, matrixID) )


# 8 bit integer conversion
tsframe_max = tsframe.max()
tsframe_8U = cv2.convertScaleAbs(tsframe, alpha=(255.0/tsframe_max))
#tsframe = cv2.imread('blob_04_noise.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
# Normalize frame
#tsframe /= max(1.0, frameManager.get_max_matrix(frameID, matrixID))






####################
# Spatial Filtering
####################

# OpennCV border types
#cv2.BORDER_CONSTANT,
#cv2.BORDER_REPLICATE, 
#cv2.BORDER_REFLECT, 
#cv2.BORDER_REFLECT_101]

d = 3 # Kernel diameter
cutoff = 0.001  

#----------------------------------------
# Morphological Transformation: Opening
#----------------------------------------
kernel_opening = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
opening = cv2.morphologyEx(tsframe, cv2.MORPH_OPEN, kernel_opening) 
masked = ma.masked_greater(opening, 0) # Masked where result of filter > 0
opening_masked = np.copy(tsframe)
opening_masked[~masked.mask] = 0

#-------------
# Box filter
#-------------
box = cv2.blur(tsframe,(d,d), borderType=cv2.BORDER_REPLICATE) 

#-------------
 # Gaussian
#-------------
sigma = 0.85
gaussian = cv2.GaussianBlur(tsframe, (d,d), sigma, borderType=cv2.BORDER_REPLICATE)

#---------------
# Median filter
#---------------
median = cv2.medianBlur(tsframe, d) 
masked = ma.masked_greater(median, 0) # Masked where result of filter > 0
median_masked = np.copy(tsframe)
median_masked[~masked.mask] = 0


#------------------
# Bilateral filter
#------------------
sigma_color = 10000 # Sigma color (Bilateral filter)
sigma_space = 10 # Sigma space (Bilateral filter)
bilateral = cv2.bilateralFilter(tsframe, d, sigma_color, sigma_space, borderType=cv2.BORDER_REPLICATE) 

#------------------
# Non local means
#------------------
nlm_h = 50
non_local_means_8U = cv2.fastNlMeansDenoising(tsframe_8U, None, h=nlm_h, templateWindowSize=d, searchWindowSize=d)
non_local_means = non_local_means_8U.astype(np.float32) * (tsframe_max/255.0)




#####################
# Temporal Filtering
#####################

#-------------------------
# Per taxel Kalman filter
#-------------------------
from pykalman import KalmanFilter
tsframes3D_kalman = np.empty((height, width, numTSFrames)) # height, width, depth
for x in range(width):
    for y in range(height):
        # Kalman filter
        transition_matrix = 1.0 # F
        transition_offset = 0.0 # b
        observation_matrix = 1.0 # H
        observation_offset = 0.0 # d
        transition_covariance = 0.05 # Q
        observation_covariance = 0.5  # R

        initial_state_mean = 0.0
        initial_state_covariance = 1.0

        # sample from model
        kf = KalmanFilter(
                transition_matrix, observation_matrix, transition_covariance,
                observation_covariance, transition_offset, observation_offset,
                initial_state_mean, initial_state_covariance,
            )

        #kf = kf.em(y, n_iter=5)
        taxels = tsframes3D[y,x,:]
        tsframes3D_kalman[y,x,:] = np.round( kf.filter(taxels)[0].flatten() )

kalman = tsframes3D_kalman[:,:,frameID].astype(np.float32)


############################
# Spatio-Temporal Filtering
############################

#---------------
# Kalman + Median filter
#---------------
kalman_median = cv2.medianBlur(kalman, d) 
masked = ma.masked_greater(kalman_median, 0) # Masked where result of filter > 0
kalman_median_masked = np.copy(tsframe)
kalman_median_masked[~masked.mask] = 0


#------------------
# 3D Median filter
#------------------
from scipy import ndimage

#median3D = ndimage.median_filter(tsframes3D[:,:,frameID], size=(3,3), mode='nearest') # Seems to be equivalent to OpenCV
median3D = ndimage.median_filter(tsframes3D[:,:,frameID-1:frameID], size=(3,3,2), mode='nearest')[:,:,0]
masked = ma.masked_greater(median3D, 0) # Masked where result of filter > 0
median3D_masked = np.copy(tsframe)
median3D_masked[~masked.mask] = 0

#------------------
# Minimum 3D filter
#------------------
minimum3D = ndimage.maximum_filter(tsframes3D[:,:,frameID-1:frameID], size=(0,0,2), mode='nearest')[:,:,0]
masked = ma.masked_greater(minimum3D, 0) # Masked where result of filter > 0
minimum3D_masked = np.copy(tsframe)
minimum3D_masked[~masked.mask] = 0

#----------------------------
# Minimum 3D filter + Median
#----------------------------
minimum3D = ndimage.minimum_filter(tsframes3D[:,:,frameID-1:frameID], size=(0,0,2), mode='nearest')[:,:,0]
minimum3D_median = cv2.medianBlur(minimum3D.astype(np.float32), d) 
masked = ma.masked_greater(minimum3D_median, 0) # Masked where result of filter > 0
minimum3D_median_masked = np.copy(tsframe)
minimum3D_median_masked[~masked.mask] = 0

#----------------------------
# 3D Total variation
#----------------------------
from skimage.restoration import denoise_tv_chambolle
#tv3D = denoise_tv_chambolle(tsframes3D[:,:,frameID-1:frameID], weight=0.2, multichannel=True)[:,:,0]
profile_max = np.max(tsframes3D[:,:,:])
tsframes3D_normalized = tsframes3D[:,:,:] / profile_max
tv3D = denoise_tv_chambolle(tsframes3D_normalized[:,:,frameID-1:frameID], weight=0.5, multichannel=True)[:,:,0]
tv3D = tv3D * profile_max

masked = ma.masked_greater(tv3D, 0) # Masked where result of filter > 0
tv3D_masked = np.copy(tsframe)
tv3D_masked[~masked.mask] = 0

# 1D Total variation
#averages2 = np.reshape(averages, (-1, 1))
#tv3D = denoise_tv_chambolle(averages2, weight=0.5, multichannel=False)[:,:,0]



############
# Plotting
############
#colormap=plt.get_cmap('gray')
colormap=plt.get_cmap('YlOrRd_r')
#colormap = plt.get_cmap('afmhot')
#colormap.set_under([0.0, 0.0, 0.0])
colormap.set_under([0.2, 0.2, 0.2])
#colormap.set_under([1.0, 1.0, 1.0])

text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0

size_factor = 1.0
figure_width = size_factor*text_width
figure_height = (figure_width / golden_ratio)
#figure_height = 1.3 * figure_width
figure_size = [figure_width, figure_height]

config.load_config_small()


def plot_filter(ax, name, tsframe, tsframe_max):
    width = tsframe.shape[1]; height = tsframe.shape[0]
    xs,ys = np.meshgrid(np.arange(0, width+1), np.arange(0, height+1))
    # Workaround inverted y-axis
    ax.invert_yaxis()

    # pcolormesh aligns cells on their edges, while imshow aligns them on their centers.
    ax.pcolormesh(xs-0.5, ys-0.5, tsframe, cmap=colormap, vmin=1.0, vmax=tsframe_max,
                  shading="faceted", linestyle="dashed", linewidth=0.5, edgecolor=[0.0, 0.0, 0.0])

    # Absolute number
    for i,j in ((x,y) for x in np.arange(0, len(tsframe))
        for y in np.arange(0, len(tsframe[0]))):
            if tsframe[i][j] >= 1:
                ax.annotate(str(int(tsframe[i][j])), xy=(j,i), fontsize=2.0, ha='center', va='center')

    ax.set_aspect('equal')
    ax.set_xlim([-0.5, width-0.5])
    ax.set_ylim([height-0.5, -0.5])

    ax.xaxis.tick_top()
    ax.tick_params(axis='both', which='both', left='off', right='off', bottom='off', top='off', labeltop='on')
    #ax.xaxis.set_major_locator(plt.NullLocator())
    #ax.yaxis.set_major_locator(plt.NullLocator())
    
    ax.set_title(name, fontsize=12)
    ax.title.set_y(1.10)
    #ax.title.set_y(1.03)
    




##################
# Spatial filters
##################
fig, axes = plt.subplots(nrows=1, ncols=8, sharex=False, sharey=False, squeeze=True, figsize=figure_size, dpi=100)


plot_filter(axes[0], "Original", tsframe, tsframe_max)
plot_filter(axes[1], "Opening", opening, tsframe_max)
plot_filter(axes[2], "Box", box, tsframe_max) # a.k.a. Uniform filter
plot_filter(axes[3], "Gaussian", gaussian, tsframe_max)
plot_filter(axes[4], "Median", median, tsframe_max)
plot_filter(axes[5], "NLM", non_local_means, tsframe_max)
plot_filter(axes[6], "tv3D", tv3D, tsframe_max)
plot_filter(axes[7], "Bilateral", bilateral, tsframe_max)

# Adjust margins and padding of entire plot
#plt.subplots_adjust(top=0.88, left = 0.07, bottom=0.0, right = 1.0)  # Legend on top
#plt.subplots_adjust(wspace=0.2, hspace=0.0)

fig.tight_layout()
#fig.show()

plotname = "spatial_filtering"
fig.savefig(plotname+".pdf", pad_inches=0, bbox_inches='tight', dpi=fig.dpi) # pdf
#fig.savefig(plotname+".pgf", pad_inches=0, bbox_inches='tight', dpi=fig.dpi) # pgf






###########
# Ghosting
###########

def plot_filter_ghosting(ax, name, tsframe, tsframe_max):
    width = tsframe.shape[1]; height = tsframe.shape[0]
    xs,ys = np.meshgrid(np.arange(0, width+1), np.arange(0, height+1))
    # Workaround inverted y-axis
    ax.invert_yaxis()

    # pcolormesh aligns cells on their edges, while imshow aligns them on their centers.
    ax.pcolormesh(xs-0.5, ys-0.5, tsframe, cmap=colormap, vmin=1.0, vmax=tsframe_max,
                  shading="faceted", linestyle="dashed", linewidth=0.5, edgecolor=[0.0, 0.0, 0.0])

    # Absolute number
    for i,j in ((x,y) for x in np.arange(0, len(tsframe))
        for y in np.arange(0, len(tsframe[0]))):
            if tsframe[i][j] >= 1:
                ax.annotate(str(int(tsframe[i][j])), xy=(j,i), fontsize=3, ha='center', va='center')

    ax.set_aspect('equal')
    ax.set_xlim([-0.5, width-0.5])
    ax.set_ylim([height-0.5, -0.5])

    ax.xaxis.tick_top()
    ax.tick_params(axis='both', which='both', left='off', right='off', bottom='off', top='off', labeltop='on')
    #ax.xaxis.set_major_locator(plt.NullLocator())
    #ax.yaxis.set_major_locator(plt.NullLocator())
    
    ax.set_title(name, fontsize=8)
    ax.title.set_y(1.08)

    
    
text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0
size_factor = 0.75
figure_width = size_factor*text_width
figure_height = figure_width
figure_size = [figure_width, figure_height]

fig, axes = plt.subplots(nrows=2, ncols=4, sharex=False, sharey=False, squeeze=True, figsize=figure_size, dpi=100)


plot_filter_ghosting(axes[0][0], "Original", tsframe, tsframe_max)
plot_filter_ghosting(axes[0][1], "Opening", opening_masked, tsframe_max)
plot_filter_ghosting(axes[0][2], "Median", median_masked, tsframe_max)
plot_filter_ghosting(axes[0][3], "Median 3D", median3D_masked, tsframe_max)
plot_filter_ghosting(axes[1][0], "Kalman", kalman, tsframe_max)
plot_filter_ghosting(axes[1][1], "Kalman + Median", kalman_median_masked, tsframe_max)
plot_filter_ghosting(axes[1][2], "Minimum", minimum3D_masked, tsframe_max)
plot_filter_ghosting(axes[1][3], "Minimum + Median", minimum3D_median_masked, tsframe_max)


# Adjust margins and padding of entire plot
plt.subplots_adjust(top=0.85, left = 0.07, bottom=0.08, right = 0.95)  # Legend on top
plt.subplots_adjust(wspace=0.1, hspace=0.4)

#fig.tight_layout()
#fig.show()

plotname = "spatial_filtering_ghosting"
fig.savefig(plotname+".pdf", pad_inches=0, bbox_inches='tight', dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, bbox_inches='tight', dpi=fig.dpi) # pgf





#################################
# Ghosting Median 2D and 3D only
#################################

def plot_filter_ghosting_median(ax, name, tsframe, tsframe_max):
    width = tsframe.shape[1]; height = tsframe.shape[0]
    xs,ys = np.meshgrid(np.arange(0, width+1), np.arange(0, height+1))
    # Workaround inverted y-axis
    ax.invert_yaxis()

    # pcolormesh aligns cells on their edges, while imshow aligns them on their centers.
    ax.pcolormesh(xs-0.5, ys-0.5, tsframe, cmap=colormap, vmin=1.0, vmax=tsframe_max,
                  shading="faceted", linestyle="dashed", linewidth=0.5, edgecolor=[0.0, 0.0, 0.0])

    # Absolute number
    for i,j in ((x,y) for x in np.arange(0, len(tsframe))
        for y in np.arange(0, len(tsframe[0]))):
            if tsframe[i][j] >= 1:
                ax.annotate(str(int(tsframe[i][j])), xy=(j,i), fontsize=4, ha='center', va='center')

    ax.set_aspect('equal')
    ax.set_xlim([-0.5, width-0.5])
    ax.set_ylim([height-0.5, -0.5])

    ax.xaxis.tick_top()
    ax.tick_params(axis='both', which='both', left='off', right='off', bottom='off', top='off', labeltop='on')
    #ax.xaxis.set_major_locator(plt.NullLocator())
    #ax.yaxis.set_major_locator(plt.NullLocator())
    
    ax.set_title(name, fontsize=12)
    ax.title.set_y(1.08)

    

    
text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0
size_factor = 0.75
figure_width = size_factor*text_width
figure_height = figure_width
figure_size = [figure_width, figure_height]

text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0

config.load_config_medium()


fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, squeeze=True, figsize=figure_size, dpi=100)


plot_filter_ghosting_median(axes[0], "Original", tsframe, tsframe_max)
plot_filter_ghosting_median(axes[1], "Median 2D", median_masked, tsframe_max)
plot_filter_ghosting_median(axes[2], "Median 3D", median3D_masked, tsframe_max)


# Adjust margins and padding of entire plot
plt.subplots_adjust(top=0.85, left = 0.07, bottom=0.08, right = 0.95)  # Legend on top
plt.subplots_adjust(wspace=0.5, hspace=0.0)

#fig.tight_layout()
#fig.show()

plotname = "spatial_filtering_ghosting_median"
fig.savefig(plotname+".pdf", pad_inches=0, bbox_inches='tight', dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, bbox_inches='tight', dpi=fig.dpi) # pgf


