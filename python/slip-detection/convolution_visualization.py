# -*- coding: utf-8 -*-

# Load configuration file (before pyplot)
import os, sys
config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)
import configuration as config

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal 
import cv2
from scipy import ndimage
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
#from scipy.ndimage.fourier import fourier_shift


import matplotlib.patches as patches

# Custom libraries
print("CWD: " + os.getcwd() )
lib_path = os.path.abspath('../../lib')
sys.path.append(lib_path)
import framemanager_python



def loadFrame(frameManager, frameID, matrixID):
    tsframe = np.copy( frameManager.get_filtered_tsframe(frameID, matrixID) );
    # Normalize frame
    #tsframe /= max(1.0, frameManager.get_max_matrix(frameID, matrixID))
    return tsframe

# Load pressure profile
profileName = os.path.abspath("slip_test2_001135-001323.dsa")
frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileName)
frameManager.set_filter_median(1, True)
numFrames = frameManager.get_tsframe_count()
matrixID = 1
frame0 = loadFrame(frameManager, 61, matrixID)
frame1 = loadFrame(frameManager, 81, matrixID)
frame2 = loadFrame(frameManager, 117, matrixID)



##########################################################
# Slip-detection: tracking centroid of convolution matrix
##########################################################

# Frame dimensions
cols = frame0.shape[1]
rows = frame0.shape[0]

# Convolution matrix dimensions 
cols_C = 2*cols-1
rows_C = 2*rows-1


# Indices of corresponding taxel position in C
A = np.tile(np.arange(1.0, cols_C+1), (cols_C, 1)) - (cols_C+1)/2 # Repeat rows and substract zeroing offset
B = np.tile(np.arange(1.0, rows_C+1), (rows_C, 1)).T - (rows_C+1)/2 # Repeat columns and substract zeroing offset


########################################################
## Convolution of frame0 with frame1
########################################################
C_1 = scipy.signal.convolve2d(frame0, frame1, mode='full', boundary='fill', fillvalue=0)

means_columns= np.mean(C_1, 0, keepdims=True) # Means of columns (along y-axis)
means_rows = np.mean(C_1, 1, keepdims=True) # Means of rows (along x-axis) 

shift_x = np.mean( (np.dot(A,means_columns.T)) / np.sum(means_columns) ) # np.dot performs matrix multiplication
shift_y = np.mean( (np.dot(means_rows.T, B)) / np.sum(means_rows) ) # np.dot performs matrix multiplication

displacement1 = np.array([shift_x, shift_y])


########################################################
## Convolution of frame1 with frame2
########################################################
C_2 = scipy.signal.convolve2d(frame1, frame2, mode='full', boundary='fill', fillvalue=0)

means_columns = np.mean(C_2, 0, keepdims=True) # Means of columns (along y-axis)
means_rows = np.mean(C_2, 1, keepdims=True) # Means of rows (along x-axis) 

shift_x = np.mean( (np.dot(A,means_columns.T)) / np.sum(means_columns) ) # np.dot performs matrix multiplication
shift_y = np.mean( (np.dot(means_rows.T, B)) / np.sum(means_rows) ) # np.dot performs matrix multiplication

displacement2 = np.array([shift_x, shift_y])

# Final slip vector 
slipvector = displacement2 - displacement1

centroid1 = ndimage.measurements.center_of_mass(frame0)
centroid2 = ndimage.measurements.center_of_mass(frame2)
slipvector2 = np.asarray(centroid2) - np.asarray(centroid1)
correlation1 = scipy.signal.correlate(frame0, frame1, mode='full')
correlation2 = scipy.signal.correlate(frame1, frame2, mode='full')




'''
#############################
# skimage phase correlation
#############################
# Add border padding due to strange behavior of cv::findContours() in case of contours touching the image border
padding = 0
frame1_padded = cv2.copyMakeBorder(frame1, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0) )
frame2_padded = cv2.copyMakeBorder(frame2, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0) )

# pixel precision first
slipvector_skimage, error, diffphase = register_translation(frame1_padded, frame2_padded, upsample_factor=32)

image_product = np.fft.fft2(frame1_padded) * np.fft.fft2(frame2_padded).conj()
cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
cc_image = _upsampled_dft(image_product, upsampled_region_size=50, upsample_factor=32, axis_offsets=(slipvector_skimage*32)+25).conj()
'''



'''
#############################
# Phase correlation
#############################
# http://stackoverflow.com/questions/2771021/is-there-an-image-phase-correlation-library-available-for-python
from matplotlib import pyplot
from scipy.fftpack import fftn, ifftn
frame1_scaled = scipy.ndimage.zoom(frame1_padded, 8, order=3)
frame2_scaled = scipy.ndimage.zoom(frame2_padded, 8, order=3)

M = frame2_padded.shape[0]
N = frame2_padded.shape[1]
hanning = np.outer(np.hanning(M),np.hanning(N))

corr = (ifftn(fftn(frame2_padded*hanning)*ifftn(frame1_padded*hanning))).real
pyplot.imshow(corr, cmap='gray', interpolation="nearest")
pyplot.show()
'''



############
# Plotting
############

def plot_convolution(plotname, tsframe, tsframe_max, paperwidth):
    colormap=plt.get_cmap('YlOrRd_r')
    colormap.set_under([0.2, 0.2, 0.2])

    figure_width = paperwidth
    figure_height = paperwidth
    figure_size = [figure_width, figure_height]
    config.load_config_small()

    width = tsframe.shape[1]; height = tsframe.shape[0]
    xs,ys = np.meshgrid(np.arange(0, width+1), np.arange(0, height+1))
   
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figure_size, dpi=100)
    ax = axes
    # Workaround inverted y-axis
    ax.invert_yaxis()

    # pcolormesh aligns cells on their edges, while imshow aligns them on their centers.
    ax.pcolormesh(xs-0.5, ys-0.5, tsframe, cmap=colormap, vmin=0.1, vmax=tsframe_max,
                  shading="faceted", linestyle="dashed", linewidth=0.5, edgecolor=[0.0, 0.0, 0.0])

    # Absolute number
    #for i,j in ((x,y) for x in np.arange(0, len(tsframe))
    #    for y in np.arange(0, len(tsframe[0]))):
    #        if tsframe[i][j] >= 1:
    #            ax.annotate(str(int(tsframe[i][j])), xy=(j,i), fontsize=3.5, ha='center', va='center')

    ax.set_aspect('equal')
    ax.set_xlim([-0.5, width-0.5])
    ax.set_ylim([height-0.5, -0.5])

    ax.xaxis.tick_top()
    ax.tick_params(axis='both', which='both', left='off', right='off', bottom='off', top='off', labeltop='on',
                   pad=1, labelsize=4)
    #ax.xaxis.set_major_locator(plt.NullLocator())
    #ax.yaxis.set_major_locator(plt.NullLocator())

    fig.tight_layout()

    fig.savefig(plotname+".pdf", pad_inches=0, bbox_inches='tight', dpi=fig.dpi) # pdf
    fig.savefig(plotname+".pgf", pad_inches=0, bbox_inches='tight', dpi=fig.dpi) # pgf



def plot_convolution2(plotname, tsframe, tsframe_max, paperwidth):
    colormap=plt.get_cmap('YlOrRd_r')
    colormap.set_under([0.2, 0.2, 0.2])

    figure_width = paperwidth
    figure_height = paperwidth
    figure_size = [figure_width, figure_height]
    config.load_config_small()

    width = tsframe.shape[1]; height = tsframe.shape[0]
    xs,ys = np.meshgrid(np.arange(0, width+1), np.arange(0, height+1))
   
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figure_size, dpi=100)
    ax = axes
    # Workaround inverted y-axis
    ax.invert_yaxis()

    # pcolormesh aligns cells on their edges, while imshow aligns them on their centers.
    ax.pcolormesh(xs-0.5, ys-0.5, tsframe, cmap=colormap, vmin=0.1, vmax=tsframe_max,
                  shading="faceted", linestyle="dashed", linewidth=0.5, edgecolor=[0.0, 0.0, 0.0])

    # Absolute number
    #for i,j in ((x,y) for x in np.arange(0, len(tsframe))
    #    for y in np.arange(0, len(tsframe[0]))):
    #        if tsframe[i][j] >= 1:
    #            ax.annotate(str(int(tsframe[i][j])), xy=(j,i), fontsize=3.5, ha='center', va='center')

    #find peak
    poi = np.asarray(np.unravel_index(np.argmax(tsframe), tsframe.shape))
    px = poi[1]
    py = poi[0]

    ax.plot(px, py, "o", ms=1.5, mew=0.3, mec=[0.0, 0.0, 0.0, 1.0], mfc=[0.0, 0.0, 1.0, 1.0])
    ax.plot(px-1, py, "o", ms=1.5, mew=0.3, mec=[0.0, 0.0, 0.0, 1.0], mfc=[1.0, 1.0, 1.0, 1.0])
    ax.plot(px+1, py, "o", ms=1.5, mew=0.3, mec=[0.0, 0.0, 0.0, 1.0], mfc=[1.0, 1.0, 1.0, 1.0])
    ax.plot(px, py-1, "o", ms=1.52, mew=0.3, mec=[0.0, 0.0, 0.0, 1.0], mfc=[1.0, 1.0, 1.0, 1.0])
    ax.plot(px, py+1, "o", ms=1.5, mew=0.3, mec=[0.0, 0.0, 0.0, 1.0], mfc=[1.0, 1.0, 1.0, 1.0])

    cx = (tsframe.shape[1]-1)/2
    cy = (tsframe.shape[0]-1)/2
    ax.plot(cx, cy, "x", ms=2, mew=0.3, color=[1.0, 1.0, 1.0, 1.0], mfc='None')
 
    r = 1.5
    kernel1 =  patches.Rectangle((px-r, py-r), 2*r, 2*r, lw=1.5, ec=[0.0, 0.0, 0.0, 1.0], fc=[1.0, 1.0, 1.0, 0.0] )
    kernel2 =  patches.Rectangle((px-r, py-r), 2*r, 2*r, lw=0.5, ec=[0.0, 0.0, 1.0, 1.0], fc=[0.0, 0.0, 0.0, 0.0]  )
    ax.add_patch(kernel1)
    ax.add_patch(kernel2)
    
    ax.set_aspect('equal')
    ax.set_xlim([-0.5, width-0.5])
    ax.set_ylim([height-0.5, -0.5])

    ax.xaxis.tick_top()
    ax.tick_params(axis='both', which='both', left='off', right='off', bottom='off', top='off', labeltop='on',
                   pad=1, labelsize=4)
    #ax.xaxis.set_major_locator(plt.NullLocator())
    #ax.yaxis.set_major_locator(plt.NullLocator())

    fig.tight_layout()

    fig.savefig(plotname+".pdf", pad_inches=0, bbox_inches='tight', dpi=fig.dpi) # pdf
    fig.savefig(plotname+".pgf", pad_inches=0, bbox_inches='tight', dpi=fig.dpi) # pgf



def plot_convolution3(plotname, tsframe, paperwidth):
    colormap=plt.get_cmap('YlOrRd_r')
    colormap.set_under([0.2, 0.2, 0.2])

    figure_width = paperwidth
    figure_height = paperwidth
    figure_size = [figure_width, figure_height]
    config.load_config_small()


    # find peak and scale neighbourhood
    poi = np.asarray(np.unravel_index(np.argmax(tsframe), tsframe.shape))
    px = poi[1]
    py = poi[0]

    r = 1
    scalefactor = 4.0
    peak_area = tsframe[py-r:py+r+1, px-r:px+r+1]
    peak_area = scipy.ndimage.zoom(peak_area, scalefactor, order=3)
    # Point of interest of peak_area
    poi_area = np.asarray(np.unravel_index(np.argmax(peak_area), peak_area.shape))
    px_area = poi_area[1]
    py_area = poi_area[0]
   
    offset_x_area = (scalefactor*(2*r+1)-1)/2.0 - px_area
    offset_y_area = (scalefactor*(2*r+1)-1)/2.0 - py_area

    width = peak_area.shape[1]; height = peak_area.shape[0]
    xs,ys = np.meshgrid(np.arange(0, width+1), np.arange(0, height+1))
  
  
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figure_size, dpi=100)
    ax = axes
    # Workaround inverted y-axis
    ax.invert_yaxis()

    # pcolormesh aligns cells on their edges, while imshow aligns them on their centers.
    ax.pcolormesh(xs-0.5, ys-0.5, peak_area, cmap=colormap, vmin=0.1, vmax=np.max(peak_area),
                  shading="faceted", linestyle="dashed", linewidth=0.5, edgecolor=[0.0, 0.0, 0.0])

    # Absolute number
    #for i,j in ((x,y) for x in np.arange(0, len(tsframe))
    #    for y in np.arange(0, len(tsframe[0]))):
    #        if tsframe[i][j] >= 1:
    #            ax.annotate(str(int(tsframe[i][j])), xy=(j,i), fontsize=3.5, ha='center', va='center')


    r = 3.5
    #extent1 =  patches.Rectangle((-0.48, -0.48), 11.96, 11.96, lw=1.5, ec=[0.0, 0.0, 0.0, 1.0], fc=[1.0, 1.0, 1.0, 0.0], zorder=98 )
    #extent2 =  patches.Rectangle((-0.48, -0.48), 11.96, 11.96, lw=0.5, ec=[0.0, 0.0, 1.0, 1.0], fc=[0.0, 0.0, 0.0, 0.0], zorder=99 )
    #ax.add_patch(extent1)
    #ax.add_patch(extent2)
    
    kernel1 =  patches.Rectangle((px_area-r, py_area-r), 2*r, 2*r, lw=1.5, ec=[0.0, 0.0, 0.0, 1.0], fc=[1.0, 1.0, 1.0, 0.0] )
    kernel2 =  patches.Rectangle((px_area-r, py_area-r), 2*r, 2*r, lw=0.5, ec=[0.0, 1.0, 0.0, 1.0], fc=[0.0, 0.0, 0.0, 0.0] )
    ax.add_patch(kernel1)
    ax.add_patch(kernel2)
    
    ax.set_aspect('equal')
    ax.set_xlim([-0.5, width-0.5])
    ax.set_ylim([height-0.5, -0.5])


    #ax.plot(px_area, py_area, "x", ms=2, mew=0.3, color=[1.0, 1.0, 1.0, 1.0], mfc='None')
    ax.plot(px_area, py_area, "o", ms=1.5, mew=0.3, mec=[0.0, 0.0, 0.0, 1.0], mfc=[0.0, 1.0, 0.0, 1.0])
    ax.plot(px_area+offset_x_area, py_area+offset_y_area, "o", ms=1.5, mew=0.3, mec=[0.0, 0.0, 0.0, 1.0], mfc=[0.0, 0.0, 1.0, 1.0])
    


    ax.xaxis.tick_top()
    ax.tick_params(axis='both', which='both', left='off', right='off', bottom='off', top='off', labeltop='on',
                   pad=1, labelsize=4)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    fig.tight_layout()

    fig.savefig(plotname+".pdf", pad_inches=0, bbox_inches='tight', dpi=fig.dpi) # pdf
    fig.savefig(plotname+".pgf", pad_inches=0, bbox_inches='tight', dpi=fig.dpi) # pgf





max_val_frame = np.max([np.max(frame0), np.max(frame1), np.max(frame2)]);
max_val_convolution = np.max([np.max(C_1), np.max(C_2)]);

plot_convolution("convolution_frame_0", frame0, max_val_frame, 1.5/2.54 + 0.25)
plot_convolution("convolution_frame_1", frame1, max_val_frame, 1.5/2.54 + 0.25)
plot_convolution("convolution_frame_2", frame2, max_val_frame, 1.5/2.54 + 0.25)
plot_convolution("convolution_C_1", C_1, max_val_convolution, 3/2.54 + 0.25)
plot_convolution("convolution_C_2", C_2, max_val_convolution, 3/2.54 + 0.25)

plot_convolution("correlation_1", correlation1, max_val_convolution, 3/2.54 + 0.25)

plot_convolution2("correlation_2", correlation2, max_val_convolution, 3/2.54 + 0.25)
plot_convolution3("correlation_2_interpolated", correlation2,  0.475*3/2.54 + 0.25) 
