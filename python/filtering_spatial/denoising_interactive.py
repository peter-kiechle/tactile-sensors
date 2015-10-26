# -*- coding: utf-8 -*-
 
################################
# Interactive filtering test
################################
 
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy.ma as ma


import cv2

def update(val):
    
    # Get parameter
    vmin = slider_cutoff.val
    borderType = borderTypes[int(np.around(slider_bordertype.val)) ] # Border type
    d = int(np.around(slider_diameter.val)) # Diameter
    sigma = slider_sigma.val # Sigma
    sigma_color = slider_sigma_color.val # Sigma color (Bilateral filter)
    sigma_space = slider_sigma_space.val # Sigma space (Bilateral filter)
    nlm_h = slider_nlm.val
    
    # Apply filters
    box = cv2.blur(img, (d,d), borderType=borderType) # Box filter
    gaussian = cv2.GaussianBlur(img, (d,d), sigma, borderType=borderType) # Gaussian
    median = cv2.medianBlur(img, d) # Median 
    bilateral = cv2.bilateralFilter(img, d, sigma_color, sigma_space, borderType=borderType) # Bilateral   
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_CROSS,(d,d))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_opening) 
    masked = ma.masked_greater(opening, 0)
    opening2 = np.copy(img)
    opening2[~masked.mask] = 0
    non_local_means = cv2.fastNlMeansDenoising(img, None, h=nlm_h, templateWindowSize=d, searchWindowSize=d)
    
    # Show results
    original.set_clim(vmin, 255.0)
    box_filter.set_data(box)
    box_filter.set_clim(vmin, 255.0)
    gaussian_filter.set_data(gaussian)   
    gaussian_filter.set_clim(vmin, 255.0)
    median_filter.set_data(median)
    median_filter.set_clim(vmin, 255.0)
    bilateral_filter.set_data(bilateral)
    bilateral_filter.set_clim(vmin, 255.0)
    opening_filter.set_data(opening)
    opening_filter.set_clim(vmin, 255.0)
    opening_filter2.set_data(opening2)
    opening_filter2.set_clim(vmin, 255.0)
    nlm_filter.set_data(non_local_means)
    nlm_filter.set_clim(vmin, 255.0)
    
    
    
    
img = cv2.imread('blob_04_noise.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)

d = 3 # Diameter

# cv2.BORDER_WRAP (3L) does not work
borderTypes = [cv2.BORDER_CONSTANT,
               cv2.BORDER_REPLICATE, 
               cv2.BORDER_REFLECT, 
               cv2.BORDER_REFLECT_101]
               
cutoff = 0.001             
borderType = 0
sigma = 0.85 # Sigma
sigma_color = 50 # Sigma color (Bilateral filter)
sigma_space = 50 # Sigma space (Bilateral filter)
nlm_h = 50

#-------------
# Box filter
#-------------
box = cv2.blur(img,(d,d), borderType=borderType) 

#-------------
 # Gaussian
#-------------
gaussian = cv2.GaussianBlur(img, (d,d), sigma, borderType=borderType)

#---------------
# Median filter
#---------------
median = cv2.medianBlur(img, d) 

#------------------
# Bilateral filter
#------------------
bilateral = cv2.bilateralFilter(img, d, sigma_color, sigma_space, borderType=borderType) 

#----------------------------------------
# Morphological Transformation: Opening
#----------------------------------------
kernel_opening = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

# Normal opening on gray scale image
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_opening) 

# Masked where result of opening > 0
masked = ma.masked_greater(opening, 0)
opening2 = np.copy(img)
opening2[~masked.mask] = 0

#------------------
# Non local means
#------------------
non_local_means = cv2.fastNlMeansDenoising(img, None, h=nlm_h, templateWindowSize=d, searchWindowSize=d)



cdict = {'red': ((0.0, 0.9, 0.9),
                     (1.0, 0.9, 0.9)),
           'green': ((0.0, 0.9, 0.9),
                     (1.0, 0.0, 0.0)),
           'blue':  ((0.0, 0.0, 0.0),
                     (1.0, 0.0, 0.0))}
plt.register_cmap(name='YELLOW_RED', data=cdict)

colormap = plt.get_cmap('YELLOW_RED')
colormap.set_under([1.0, 1.0, 1.0])




pl.subplots_adjust(left=0.15, bottom=0.35)
#pl.subplots_adjust(left=0.25, bottom=0.25)

pl.subplot(191)
original = pl.imshow(img, cmap=colormap, vmin=cutoff, vmax=255.0, interpolation='nearest')
pl.title('Original')
pl.xticks([]), pl.yticks([])


pl.subplot(192)
box_filter = pl.imshow(box, cmap=colormap, vmin=cutoff, vmax=255.0, interpolation='nearest')
pl.title('Box filter')
pl.xticks([]), pl.yticks([])


pl.subplot(193)
gaussian_filter = pl.imshow(gaussian, cmap=colormap, vmin=cutoff, vmax=255.0, interpolation='nearest')
pl.title('Gaussian')
pl.xticks([]), pl.yticks([])


pl.subplot(194)
median_filter = pl.imshow(median, cmap=colormap, vmin=cutoff, vmax=255.0, interpolation='nearest')
pl.title('Median')
pl.xticks([]), pl.yticks([])


pl.subplot(195)
bilateral_filter = pl.imshow(bilateral, cmap=colormap, vmin=cutoff, vmax=255.0, interpolation='nearest')
pl.title('Bilateral')
pl.xticks([]), pl.yticks([])

pl.subplot(196)
opening_filter = pl.imshow(opening, cmap=colormap, vmin=cutoff, vmax=255.0, interpolation='nearest')
pl.title('Opening')
pl.xticks([]), pl.yticks([])

pl.subplot(197)
opening_filter2 = pl.imshow(opening2, cmap=colormap, vmin=cutoff, vmax=255.0, interpolation='nearest')
pl.title('Opening 2')
pl.xticks([]), pl.yticks([])

pl.subplot(198)
nlm_filter = pl.imshow(non_local_means, cmap=colormap, vmin=cutoff, vmax=255.0, interpolation='nearest')
pl.title('NLM')
pl.xticks([]), pl.yticks([])


pl.show()
#plt.savefig("spatial_filtering.pdf", pad_inches=0, dpi=100) # pdf   


slider_axis_cutoff =      pl.axes([0.25, 0.35, 0.65, 0.03])
slider_axis_bordertype =  pl.axes([0.25, 0.3, 0.65, 0.03])
slider_axis_diameter =    pl.axes([0.25, 0.25, 0.65, 0.03])
slider_axis_sigma =       pl.axes([0.25, 0.2, 0.65, 0.03])
slider_axis_sigma_color = pl.axes([0.25, 0.15, 0.65, 0.03])
slider_axis_sigma_space = pl.axes([0.25, 0.1, 0.65, 0.03])
slider_axis_nlm =         pl.axes([0.25, 0.05, 0.65, 0.03])

slider_cutoff = Slider(slider_axis_cutoff, 'Cut-off threshold', 0.001, 50.0, valfmt='%0.2f', valinit=cutoff)
slider_bordertype = Slider(slider_axis_bordertype, 'Border type', 0, 3, valfmt='%0.0f', valinit=borderType)
slider_diameter = Slider(slider_axis_diameter, 'Kernel diameter', 1, 7, valfmt='%0.0f', valinit=d)
slider_sigma = Slider(slider_axis_sigma, 'Sigma (Gaussian)', 0.05, 10.0, valfmt='%.2f', valinit=sigma)
slider_sigma_color = Slider(slider_axis_sigma_color, 'Sigma Color (Bilateral)', 0.05, 200, valfmt='%.2f', valinit=sigma_color)
slider_sigma_space = Slider(slider_axis_sigma_space, 'Sigma Space (Bilateral)', 0.05, 200, valfmt='%.2f', valinit=sigma_space)
slider_nlm = Slider(slider_axis_nlm, 'Filter strength (NLM)', 1, 200, valfmt='%.2f', valinit=nlm_h)

slider_cutoff.on_changed(update)
slider_bordertype.on_changed(update)
slider_diameter.on_changed(update)
slider_sigma.on_changed(update)
slider_sigma_color.on_changed(update)
slider_sigma_space.on_changed(update)
slider_nlm.on_changed(update)

