# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal
import scipy.misc
from scipy import ndimage
from scipy.optimize import curve_fit


import cv2 # padding
from skimage.feature import register_translation # phase correlation


# Alcazar
def normalized_cross_correlation(frame0, frame1):

    # Frame dimensions
    cols = frame0.shape[1]
    rows = frame0.shape[0]    

    cols_C = 2*cols-1
    rows_C = 2*rows-1

    # Indices of corresponding taxel position in C
    A = np.tile(np.arange(1.0, cols_C+1), (cols_C, 1)) - (cols_C+1)/2 # Repeat rows and substract zeroing offset
    B = np.tile(np.arange(1.0, rows_C+1), (rows_C, 1)).T - (rows_C+1)/2 # Repeat columns and substract zeroing offset


    ########################################################
    ## Convolution of reference frame with itself
    ########################################################

    C_stationary = scipy.signal.convolve2d(frame0, frame0, mode='full', boundary='fill', fillvalue=0)

    means_columns= np.mean(C_stationary, 0, keepdims=True) # Means of columns (along y-axis)
    means_rows = np.mean(C_stationary, 1, keepdims=True) # Means of rows (along x-axis) 

    shift_x = np.mean( (np.dot(A,means_columns.T)) / np.sum(means_columns) ) # np.dot performs matrix multiplication
    shift_y = np.mean( (np.dot(means_rows.T, B)) / np.sum(means_rows) ) # np.dot performs matrix multiplication

    displacement0 = np.array([shift_x, shift_y])

    ########################################################
    ## Convolution of reference frame with comparison frame
    ########################################################

    C_moving = scipy.signal.convolve2d(frame0, frame1, mode='full', boundary='fill', fillvalue=0)

    means_columns = np.mean(C_moving, 0, keepdims=True) # Means of columns (along y-axis)
    means_rows = np.mean(C_moving, 1, keepdims=True) # Means of rows (along x-axis) 

    shift_x = np.mean( (np.dot(A,means_columns.T)) / np.sum(means_columns) ) # np.dot performs matrix multiplication
    shift_y = np.mean( (np.dot(means_rows.T, B)) / np.sum(means_rows) ) # np.dot performs matrix multiplication

    displacement1 = np.array([shift_x, shift_y])

    slipvector = displacement1 - displacement0

    return slipvector
   
   



    
def paraboloid(data, a, b, c, d, e, f):
    x,y = data
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f


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



def esinc1D(x, x0, a, b): # shift, magnitude, scale
     return a * np.exp( -(b*(x-x0))**2 ) * np.sinc(x-x0)

#def esinc1D(x, x0, a, b): # shift, magnitude, scale
#    return  np.sinc(x-x0)

def esinc2d(data, x0, x1, a0, a1, b0, b1): # shift, magnitude, scale
    x,y = data
    return esinc1D(x, x0, a0, b0) * esinc1D(x, x1, a1, b1)
   

def lanczos1D(x, x0, k, s): # shift, kernel size, scale
    return s*np.sinc(x-x0) * np.sinc((x-x0)/k)
    #if x-x0 < np.abs(k):
    #    return s*np.sinc(x-x0) * np.sinc((x-x0)/k)
    #else:
    #    return 0.0

def lanczos2D(data, x0, y0, kx, ky, s): # shift, kernel size, scale
    x,y = data
    return lanczos1D(x, x0, kx, s) * lanczos1D(y, y0, ky, s)
  

'''
r = 5
#xxx = np.arange(-r,r+1,1)
xxx = np.linspace(-1, 1, 2*r+1)
yyy = np.empty(r*2+1)
for i, value in np.ndenumerate(xxx):
    yyy[i] = lanczos1D(value, 0.0, 5.0, 75.0)
'''



# All in one (upscaling + paraboloid curve fit for now)
def normalized_cross_correlation2(frame0, frame1):
    #frame0 = frame0.astype(np.float64, copy=True)
    #frame1 = frame1.astype(np.float64, copy=True)
    
    # Normalize matrices    

    #active_cells0 = np.ma.masked_where(frame0 == 0, frame0) # Restrict to nonzero values
    #mean0 = np.ma.mean(active_cells0)
    #stddev0 = np.ma.std(active_cells0)
    mean0 = np.mean(frame0)
    stddev0 = np.std(frame0)
    frame0_normalized = (frame0-mean0) / stddev0
    
    #active_cells1 = np.ma.masked_where(frame1 == 0, frame1) # Restrict to nonzero values
    #mean1 = np.ma.mean(active_cells1)
    #stddev1 = np.ma.std(active_cells1)
    mean1 = np.mean(frame1)
    stddev1 = np.std(frame1)
    frame1_normalized = (frame1-mean1) / stddev1
    
    # Upscaling
    # Order
    # 0: Nearest-neighbor
    # 1: Bi-linear (default)
    # 2: Bi-quadratic
    # 3: Bi-cubic
    # 4: Bi-quartic
    # 5: Bi-quintic
    #frame0_normalized = scipy.ndimage.zoom(frame0_normalized, 2, order=2)
    #frame1_normalized = scipy.ndimage.zoom(frame1_normalized, 2, order=2)
    
    # New matrix dimensions
    #cols = frame0_normalized.shape[1]
    #rows = frame0_normalized.shape[0] 
    
    # Correlation
    correlation = scipy.signal.correlate(frame0_normalized, frame1_normalized, mode='full')

    # Point of interest
    peak = np.asarray(np.unravel_index(np.argmax(correlation), correlation.shape))
    px = peak[1]
    py = peak[0]
    offset_x = (correlation.shape[1]-1)/2 - px
    offset_y = (correlation.shape[0]-1)/2 - py
   
    # Just the integer position of the peak
    #slipvector_integer= np.array([offset_x, offset_y])
    #return slipvector_integer

    #-------------------------------------------------------------------------
    # Subpixel accuracy by upscaling + interpolation
    #-------------------------------------------------------------------------
    r = 2
    scalefactor = 4.0
    peak_area = correlation[py-r:py+r+1, px-r:px+r+1]
    peak_area = scipy.ndimage.zoom(peak_area, scalefactor, order=3)
    # Point of interest of peak_area
    poi_area = np.asarray(np.unravel_index(np.argmax(peak_area), peak_area.shape))
    px_area = poi_area[1]
    py_area = poi_area[0]
    offset_x_area = (scalefactor*(2*r+1)-1)/2.0 - px_area
    offset_y_area = (scalefactor*(2*r+1)-1)/2.0 - py_area

    #-------------------------------------------------------------------------
    # Subpixel accuracy using paraboloid curve fit of (upscaled) neighbourhood around the peak
    #-------------------------------------------------------------------------
    # Slice of neighbourhood around peak
    # The coordinate of the initial peak is assumed to be (0,0) and the width of a pixel is 1
    r = 5
    zs2D = peak_area[py_area-r:py_area+r+1, px_area-r:px_area+r+1]
    #zs2D = correlation[py-r:py+r+1, px-r:px+r+1] # Without upscaling
    xs2D = np.tile(np.arange(0.0, 2*r+1.0), (2*r+1, 1)) - r  # Repeat rows and substract zeroing offset
    ys2D = xs2D.T
    xs = np.ravel(xs2D) # Flatten arrays
    ys = np.ravel(ys2D)
    zs = np.ravel(zs2D)
    xdata = np.vstack((xs, ys))
    
    # Compute Gaussian weights
    d = np.sqrt(xs2D**2 + ys2D**2) # Distance from poi
    k = r*20
    weights = 1/np.exp(-d**d / k**2)
    
    #print offset_x_area, offset_y_area, zs2D.shape[0], zs2D.shape[1]
    
    # Least-squares fit
    model, _ = curve_fit(paraboloid, xdata, zs, sigma=np.ravel(weights) )
    #model, _ = curve_fit(paraboloid, xdata, zs)
    
    # Maximum of paraboloid (relative to peak)
    dx = (2*model[1]*model[3] - model[2]*model[4]) / (model[2]**2 - 4*model[0]*model[1])
    dy = (2*model[0]*model[4] - model[2]*model[3]) / (model[2]**2 - 4*model[0]*model[1])

    #slipvector_paraboloid = np.array([offset_x-dx, offset_y-dy]) # Without upscaling
    
    slipvector_paraboloid = np.array([offset_x + (offset_x_area - dx)/scalefactor, 
                                  offset_y + (offset_y_area - dy)/scalefactor])
    
    return slipvector_paraboloid

    
    '''
    #-------------------------------------------------------------------------
    # Subpixel accuracy using lanczos window curve fit of (upscaled) neighbourhood around the peak
    #-------------------------------------------------------------------------
    # Slice of neighbourhood around peak
    # The coordinate of the initial peak is assumed to be (0,0) and the width of a pixel is 1
    r = 5
    zs2D = peak_area[py_area-r:py_area+r+1, px_area-r:px_area+r+1]
    #zs2D = correlation[py-r:py+r+1, px-r:px+r+1] # Without upscaling
    xs2D = np.tile(np.arange(0.0, 2*r+1.0), (2*r+1, 1)) - r  # Repeat rows and substract zeroing offset
    ys2D = xs2D.T
    xs = np.ravel(xs2D) # Flatten arrays
    ys = np.ravel(ys2D)
    zs = np.ravel(zs2D)
    xdata = np.vstack((xs/r, ys/r))
    
    # xo, yo, kx, ky, s
    initial_guess = (0, 0, r, r, peak_area[py_area,px_area])

    # Least-squares fit
    model, _ = curve_fit(lanczos2D, xdata, zs, p0=initial_guess)

    dx = model[0]
    dy = model[1]

    #slipvector_curve_fit = np.array([offset_x-dx, offset_y-dy]) # Without upscaling
    slipvector_curve_fit = np.array([offset_x + (offset_x_area - dx)/scalefactor, 
                                 offset_y + (offset_y_area - dy)/scalefactor])

    
    return slipvector_curve_fit
    '''
    
    
    
    
    '''
    #-------------------------------------------------------------------------
    # Subpixel accuracy using 3-points
    #-------------------------------------------------------------------------
    
    # Gaussian (attention negative values)
    correlation = correlation + np.abs(np.min(correlation))
    nominator_x = np.log(correlation[py,px-1]) - np.log(correlation[py,px+1])
    denominator_x = 2*np.log(correlation[py,px+1]) - 4*np.log(correlation[py,px]) +2*np.log(correlation[py,px-1])
    nominator_y = np.log(correlation[py-1,px]) - np.log(correlation[py+1,px])
    denominator_y = 2*np.log(correlation[py+1,px]) - 4*np.log(correlation[py,px]) +2*np.log(correlation[py-1,px])
    
    # Parabola
    #nominator_x = correlation[py,px-1] - correlation[py,px+1]
    #denominator_x = 2*correlation[py,px-1] - 4*correlation[py,px] + 2*correlation[py,px+1]
    #nominator_y = correlation[py-1,px] - correlation[py+1,px]
    #denominator_y = 2*correlation[py-1,px] - 4*correlation[py,px] + 2*correlation[py+1,px]
    
    
    dx = nominator_x/denominator_x
    dy = nominator_y/denominator_y
    slipvector_3point = np.array([offset_x-dx, offset_y-dy])
    return slipvector_3point
    '''




# 3-points
def normalized_cross_correlation3(frame0, frame1):

    # Normalize matrices    
    mean0 = np.mean(frame0)
    stddev0 = np.std(frame0)
    frame0_normalized = (frame0-mean0) / stddev0
    
    mean1 = np.mean(frame1)
    stddev1 = np.std(frame1)
    frame1_normalized = (frame1-mean1) / stddev1

    # Correlation
    correlation = scipy.signal.correlate(frame0_normalized, frame1_normalized, mode='full')

    # Point of interest
    peak = np.asarray(np.unravel_index(np.argmax(correlation), correlation.shape))
    px = peak[1]
    py = peak[0]
    offset_x = (correlation.shape[1]-1)/2 - px
    offset_y = (correlation.shape[0]-1)/2 - py
 
    #-------------------------------------------------------------------------
    # Subpixel accuracy using 3-points
    #-------------------------------------------------------------------------
    # Gaussian (attention negative values)
    correlation = correlation + np.abs(np.min(correlation))
    nominator_x = np.log(correlation[py,px-1]) - np.log(correlation[py,px+1])
    denominator_x = 2*np.log(correlation[py,px+1]) - 4*np.log(correlation[py,px]) +2*np.log(correlation[py,px-1])
    nominator_y = np.log(correlation[py-1,px]) - np.log(correlation[py+1,px])
    denominator_y = 2*np.log(correlation[py+1,px]) - 4*np.log(correlation[py,px]) +2*np.log(correlation[py-1,px])
    
    # Parabola
    #nominator_x = correlation[py,px-1] - correlation[py,px+1]
    #denominator_x = 2*correlation[py,px-1] - 4*correlation[py,px] + 2*correlation[py,px+1]
    #nominator_y = correlation[py-1,px] - correlation[py+1,px]
    #denominator_y = 2*correlation[py-1,px] - 4*correlation[py,px] + 2*correlation[py+1,px]

    dx = nominator_x/denominator_x
    dy = nominator_y/denominator_y
    slipvector_3point = np.array([offset_x-dx, offset_y-dy])
    return slipvector_3point
 


# Phase correlation + upsampled DFT
def normalized_cross_correlation4(frame0, frame1):
    # Add border padding
    frame0_padded = cv2.copyMakeBorder(frame0, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0) )
    frame1_padded = cv2.copyMakeBorder(frame1, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0) )
    slipvector_skimage, error, diffphase = register_translation(frame1_padded, frame0_padded, upsample_factor=32)
    return slipvector_skimage[::-1]





# Intensity weighted centroid / center of gravity
def normalized_cross_correlation5(frame0, frame1):
    centroid1 = ndimage.measurements.center_of_mass(frame0)
    centroid2 = ndimage.measurements.center_of_mass(frame1)
    slipvector_centroid = np.asarray(centroid2) - np.asarray(centroid1)
    return slipvector_centroid[::-1]
    
    