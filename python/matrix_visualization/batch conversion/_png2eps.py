#!/usr/bin/python

# Convert png to eps

import sys, os
import pylab as plt
import cv2

inputfile = str(sys.argv[1])
img = cv2.imread(inputfile, cv2.CV_LOAD_IMAGE_GRAYSCALE)

# Custom colormap
cdict = {'red': ((0.0, 0.9, 0.9),
                     (1.0, 0.9, 0.9)),
         'green': ((0.0, 0.9, 0.9),
                     (1.0, 0.0, 0.0)),
         'blue':  ((0.0, 0.0, 0.0),
                     (1.0, 0.0, 0.0))}
#plt.register_cmap(name='YELLOW_RED', data=cdict)
#colormap = plt.get_cmap('YELLOW_RED')
colormap = plt.get_cmap('afmhot')
#colormap = plt.get_cmap('hot')
#colormap = plt.get_cmap('YlOrRd_r')
#colormap = plt.get_cmap('Reds_r')
#colormap = plt.get_cmap('OrRd_r')

# Shift inactive cells down to lower_color_threshold
#lower_color_threshold = -10
#img_int32 = np.array(img, dtype=int)
#img_int32[img_int32==0] = lower_color_threshold-1
#colormap.set_under([0.0, 0.0, 0.0])

ax = plt.subplot(111)

#plt.imshow(img, cmap=colormap, vmin=lower_color_threshold, vmax=255, interpolation='nearest')
plt.imshow(img, cmap=colormap, vmin=0, vmax=255, interpolation='nearest')
plt.xticks([]), plt.yticks([])
plt.tight_layout() 

basename = os.path.splitext(inputfile)[0]

plt.savefig(basename+".eps", pad_inches=0) # eps
plt.close()


