##########################################
# Load configuration file (before pyplot)
##########################################

import os, sys
config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)
import configuration as config

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal 
from scipy import ndimage
#import scipy.misc
from scipy.optimize import curve_fit

import cv2


# Custom libraries
print("CWD: " + os.getcwd() )
lib_path = os.path.abspath('../../lib')
sys.path.append(lib_path)
import framemanager_python

import module_image_moments as IM


def loadFrame(frameManager, frameID, matrixID):
    tsframe = np.copy( frameManager.get_tsframe(frameID, matrixID) );
    # Normalize frame
    #tsframe /= max(1.0, frameManager.get_max_matrix(frameID, matrixID))
    return tsframe

    

# Load pressure profile
#profileName = os.path.abspath("slip_and_rotation_teapot_handle.dsa")
profileName = os.path.abspath("slip_test2_001135-001323.dsa")
frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileName);
numFrames = frameManager.get_tsframe_count();

matrixID = 1
#frame0 = loadFrame(frameManager, 31, matrixID)
#frame1 = loadFrame(frameManager, 31, matrixID)
#frame2 = loadFrame(frameManager, 35, matrixID)
frame0 = loadFrame(frameManager, 61, matrixID)
frame1 = loadFrame(frameManager, 81, matrixID)
frame2 = loadFrame(frameManager, 117, matrixID)

frame0 = frame0.astype(np.float32, copy=True)
frame1 = frame1.astype(np.float32, copy=True)
frame2 = frame2.astype(np.float32, copy=True)


# Normalize frames
#frame0 /= max(1.0, np.max(frame0))
#frame1 /= max(1.0, np.max(frame1))
#frame2 /= max(1.0, np.max(frame2))


#------------------------------------------------------------
# Alcazar
#------------------------------------------------------------

# Frame dimensions
cols = frame0.shape[1]
rows = frame0.shape[0]    

cols_C = 2*cols-1
rows_C = 2*rows-1

# Indices of corresponding taxel position in C
A = np.tile(np.arange(1.0, cols_C+1), (cols_C, 1)) - (cols_C+1)/2 # Repeat rows and substract zeroing offset
B = np.tile(np.arange(1.0, rows_C+1), (rows_C, 1)).T - (rows_C+1)/2 # Repeat columns and substract zeroing offset

# Convolution of frame0 with frame1
convolution_1 = scipy.signal.convolve2d(frame0, frame1, mode='full', boundary='fill', fillvalue=0)

means_columns= np.mean(convolution_1, 0) # Means of columns (along y-axis)
means_rows = np.mean(convolution_1, 1) # Means of rows (along x-axis) 

shift_x = np.mean( (np.dot(A,means_columns.T)) / np.sum(means_columns) ) # np.dot performs matrix multiplication
shift_y = np.mean( (np.dot(means_rows.T, B)) / np.sum(means_rows) ) # np.dot performs matrix multiplication

displacement1 = np.array([shift_x, shift_y])


# Convolution of frame1 with frame2
convolution_2 = scipy.signal.convolve2d(frame1, frame2, mode='full', boundary='fill', fillvalue=0)

means_columns = np.mean(convolution_2, 0) # Means of columns (along y-axis)
means_rows = np.mean(convolution_2, 1) # Means of rows (along x-axis) 

shift_x = np.mean( (np.dot(A,means_columns.T)) / np.sum(means_columns) ) # np.dot performs matrix multiplication
shift_y = np.mean( (np.dot(means_rows.T, B)) / np.sum(means_rows) ) # np.dot performs matrix multiplication

displacement2 = np.array([shift_x, shift_y])

# Final slip vector 
slipvector_Alcazar = displacement2 - displacement1
#------------------------------------------------------------

#------------------------------------------------------------
# Center of gravity tracking
#------------------------------------------------------------
centroid1 = ndimage.measurements.center_of_mass(frame0)
centroid2 = ndimage.measurements.center_of_mass(frame2)
slipvector_centroid = np.asarray(centroid2) - np.asarray(centroid1)
slipvector_centroid = slipvector_centroid[::-1]
#------------------------------------------------------------




#------------------------------------------------------------
# Real NCC
#------------------------------------------------------------
#active_cells0 = np.ma.masked_where(frame0 == 0, frame0) # Restrict to nonzero values
#mean_0 = np.ma.mean(active_cells0)
#stddev_0 = np.ma.std(active_cells0)
mean_0 = np.mean(frame0)
stddev_0 = np.std(frame0)
frame0_normalized = (frame0-mean_0) / stddev_0

#active_cells1 = np.ma.masked_where(frame1 == 0, frame1) # Restrict to nonzero values
#mean_1 = np.ma.mean(active_cells1)
#stddev_1 = np.ma.std(active_cells1)
mean_1 = np.mean(frame1)
stddev_1 = np.std(frame1)
frame1_normalized = (frame1-mean_1) / stddev_1

#active_cells2 = np.ma.masked_where(frame2 == 0, frame2) # Restrict to nonzero values
#mean_2 = np.ma.mean(active_cells2)
#stddev_2 = np.ma.std(active_cells2)
mean_2 = np.mean(frame2)
stddev_2 = np.std(frame2)
frame2_normalized = (frame2-mean_2) / stddev_2

# Upscaling
# Order
# 0: Nearest-neighbor
# 1: Bi-linear (default)
# 2: Bi-quadratic
# 3: Bi-cubic
# 4: Bi-quartic
# 5: Bi-quintic
#frame0_normalized = scipy.ndimage.zoom(frame0_normalized, 2, order=3)
#frame1_normalized = scipy.ndimage.zoom(frame1_normalized, 2, order=3)
#frame2_normalized = scipy.ndimage.zoom(frame2_normalized, 2, order=3)

# New matrix dimensions
#cols = frame0_normalized.shape[1]
#rows = frame0_normalized.shape[0]    

#frame2_normalized = scipy.ndimage.zoom(frame2_normalized, 2, order=3)

# Scipy correlation
correlation_1 = scipy.signal.correlate(frame0_normalized, frame1_normalized, mode='full')
correlation_2 = scipy.signal.correlate(frame0_normalized, frame2_normalized, mode='full')

# Peak detection
poi_1 = np.asarray(np.unravel_index(np.argmax(correlation_1), correlation_1.shape))
poi_2 = np.asarray(np.unravel_index(np.argmax(correlation_2), correlation_2.shape))
px = poi_2[1]
py = poi_2[0]





#-------------------------------------------------------------------------
# Subpixel accuracy by upscaling + interpolation
#-------------------------------------------------------------------------
r = 1
scalefactor = 4.0
peak_area = correlation_2[py-r:py+r+1, px-r:px+r+1]
peak_area = scipy.ndimage.zoom(peak_area, scalefactor, order=3)
# Point of interest of peak_area
poi_area = np.asarray(np.unravel_index(np.argmax(peak_area), peak_area.shape))
px_area = poi_area[1]
py_area = poi_area[0]

offset_x = (correlation_2.shape[1]-1)/2 - px
offset_y = (correlation_2.shape[0]-1)/2 - py

offset_x_area = (scalefactor*(2*r+1)-1)/2.0 - px_area
offset_y_area = (scalefactor*(2*r+1)-1)/2.0 - py_area





#-------------------------------------------------------------------------
# Subpixel peak using 3-point curve fitting
#-------------------------------------------------------------------------

# Gaussian (attention negative values)

correlation_2b = correlation_2 + np.abs(np.min(correlation_2))
nominator_x = np.log(correlation_2b[py,px-1]) - np.log(correlation_2b[py,px+1])
denominator_x = 2*np.log(correlation_2b[py,px+1]) - 4*np.log(correlation_2b[py,px]) +2*np.log(correlation_2b[py,px-1])
nominator_y = np.log(correlation_2b[py-1,px]) - np.log(correlation_2b[py+1,px])
denominator_y = 2*np.log(correlation_2b[py+1,px]) - 4*np.log(correlation_2b[py,px]) +2*np.log(correlation_2b[py-1,px])

# Parabola
#nominator_x = correlation_2[py,px-1] - correlation_2[py,px+1]
#denominator_x = 2*correlation_2[py,px-1] - 4*correlation_2[py,px] + 2*correlation_2[py,px+1]
#nominator_y = correlation_2[py-1,px] - correlation_2[py+1,px]
#denominator_y = 2*correlation_2[py-1,px] - 4*correlation_2[py,px] + 2*correlation_2[py+1,px]

dx = nominator_x/denominator_x
dy = nominator_y/denominator_y
slipvector_3point = np.array([offset_x-dx, offset_y-dy])





#-------------------------------------------------------------------------
# Subpixel accuracy using paraboloid curve fit
#-------------------------------------------------------------------------
# Sliced neighbourhood around peak
# The coordinate of the initial peak is assumed to be (0,0) and the width of a pixel is 1
r = 5
zs2D = peak_area[py_area-r:py_area+r+1, px_area-r:px_area+r+1]
#zs2D = correlation_2[py-r:py+r+1, px-r:px+r+1]
xs2D = np.tile(np.arange(0.0, 2*r+1.0), (2*r+1, 1)) - r  # Repeat rows and substract zeroing offset
ys2D = xs2D.T
xs = np.ravel(xs2D) # Flatten arrays
ys = np.ravel(ys2D)
zs = np.ravel(zs2D)


def paraboloid(data, a, b, c, d, e, f):
    x,y = data
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

xdata = np.vstack((xs, ys))

# Compute Gaussian weights
d = np.sqrt(xs2D**2 + ys2D**2) # Distance from poi
k = r*20.0
weights = 1/np.exp(-d**d / k**2)
model, _ = curve_fit(paraboloid, xdata, zs, sigma=np.ravel(weights) )

dx = (2*model[1]*model[3] - model[2]*model[4]) / (model[2]**2 - 4*model[0]*model[1])
dy = (2*model[0]*model[4] - model[2]*model[3]) / (model[2]**2 - 4*model[0]*model[1])

#slipvector_paraboloid = np.array([offset_x-dx, offset_y+dy])

slipvector_paraboloid = np.array([offset_x + (offset_x_area - dx)/scalefactor, 
                                  offset_y + (offset_y_area - dy)/scalefactor])






#-------------------------------------------------------------------------
# Subpixel accuracy using gaussian curve fit of (upscaled) neighbourhood around the peak
#-------------------------------------------------------------------------

# http://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()


def lanczos1D(x, x0, k, s): # shift, kernel size, scale
    return s*np.sinc(x-x0) * np.sinc((x-x0)/k)
    #if x-x0 < np.abs(k):
    #    return s*np.sinc(x-x0) * np.sinc((x-x0)/k)
    #else:
    #    return 0.0

def lanczos2D(data, x0, y0, kx, ky, s): # shift, kernel size, scale
    x,y = data
    return lanczos1D(x, x0, kx, s) * lanczos1D(y, y0, ky, s)
  

# Slice of neighbourhood around peak
# The coordinate of the initial peak is assumed to be (0,0) and the width of a pixel is 1
r = 5

#zs2D = peak_area[py_area-r:py_area+r+1, px_area-r:px_area+r+1]
zs2D = correlation_2[py-r:py+r+1, px-r:px+r+1] # Without upscaling
xs2D = np.tile(np.arange(0.0, 2*r+1.0), (2*r+1, 1)) - r  # Repeat rows and substract zeroing offset
ys2D = xs2D.T
xs2D = xs2D / (2*r)
ys2D = ys2D / (2*r)
xs = np.ravel(xs2D) # Flatten arrays
ys = np.ravel(ys2D)
zs = np.ravel(zs2D)
xdata = np.vstack((xs, ys))

#max_value = peak_area[py_area, px_area]
max_value = correlation_2[py, px]

# xo, yo, kx, ky, s
initial_guess = (0, 0, r, r, max_value)


# Least-squares fit
model, _ = curve_fit(lanczos2D, xdata, zs, p0=initial_guess)

dx = model[0] * (2*r)
dy = model[1] * (2*r)

#slipvector_curve_fit = np.array([offset_x-dx, offset_y-dy]) # Without upscaling
slipvector_curve_fit = np.array([offset_x + (offset_x_area - dx)/scalefactor, 
                                 offset_y + (offset_y_area - dy)/scalefactor])

