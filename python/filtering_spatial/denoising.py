# -*- coding: utf-8 -*-

################################
# Filtering test
################################
 
import pylab as pl
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

import cv2

img_noise = cv2.imread('blob_04_noise.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)

# Morphological Transformation: Opening
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

# Normal opening on gray scale image
opening = cv2.morphologyEx(img_noise, cv2.MORPH_OPEN, kernel) 

# Non local means
non_local_means = cv2.fastNlMeansDenoising(img_noise, None, h=75, templateWindowSize=3, searchWindowSize=3)


# Masked where result of opening > 0
masked = ma.masked_greater(opening, 0)
opening2 = np.copy(img_noise)
opening2[~masked.mask] = 0

# Custom colormap
cdict = {'red': ((0.0, 0.9, 0.9),
                     (1.0, 0.9, 0.9)),
         'green': ((0.0, 0.9, 0.9),
                     (1.0, 0.0, 0.0)),
         'blue':  ((0.0, 0.0, 0.0),
                     (1.0, 0.0, 0.0))}
plt.register_cmap(name='YELLOW_RED', data=cdict)

colormap = plt.get_cmap('YELLOW_RED')
colormap.set_under([1.0, 1.0, 1.0])



ax = pl.subplot(131)

#pl.subplots_adjust(left=0.25, bottom=0.25)

pl.subplot(141)
pl.imshow(img_noise, cmap=colormap, vmin=0.0001, vmax=255.0, interpolation='nearest')
pl.title('Ghosting')
pl.xticks([]), pl.yticks([])

pl.subplot(142)
opening_filter = pl.imshow(opening, cmap=plt.get_cmap('YELLOW_RED'), vmin=0.0001, vmax=255.0, interpolation='nearest')
pl.title('Opening')
pl.xticks([]), pl.yticks([])

pl.subplot(143)
opening_filter = pl.imshow(opening2, cmap=plt.get_cmap('YELLOW_RED'), vmin=0.0001, vmax=255.0, interpolation='nearest')
pl.title('Masked opening')
pl.xticks([]), pl.yticks([])

pl.subplot(144)
opening_filter = pl.imshow(non_local_means, cmap=plt.get_cmap('YELLOW_RED'), vmin=0.0001, vmax=255.0, interpolation='nearest')
pl.title('Non local means')
pl.xticks([]), pl.yticks([])


pl.show()

