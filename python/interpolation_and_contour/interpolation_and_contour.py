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

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate


# Color map
# Define "bds_highcontrast" color map by Britton Smith <brittonsmith@gmail.com> from http://yt-project.org/ 
cdict = {'red':   ((0.0, 80/256., 80/256.),
                   (0.2, 0.0, 0.0),
                   (0.4, 0.0, 0.0),
                   (0.6, 256/256., 256/256.),
                   (0.95, 256/256., 256/256.),
                   (1.0, 150/256., 150/256.)),
         'green': ((0.0, 0/256., 0/256.),
                   (0.2, 0/256., 0/256.),
                   (0.4, 130/256., 130/256.),
                   (0.6, 256/256., 256/256.),
                   (1.0, 0.0, 0.0)),
         'blue':  ((0.0, 80/256., 80/256.),
                   (0.2, 220/256., 220/256.),
                   (0.4, 0.0, 0.0),
                   (0.6, 20/256., 20/256.),
                   (1.0, 0.0, 0.0))}

plt.register_cmap(name='bds_highcontrast', data=cdict) 

# Define YELLOW_RED colormap: each row consists of (x, y0, y1) where the x must increase from 0 to 1
#row i:    x  y0  y1
#               /
#              /
#row i+1:  x  y0  y1
cdict = {'red':   ((0.0, 0.9, 0.9),
                   (1.0, 0.9, 0.9)),
         'green': ((0.0, 0.9, 0.9),
                   (1.0, 0.0, 0.0)),
         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0))}
plt.register_cmap(name='YELLOW_RED', data=cdict) 
#cmap=plt.get_cmap('YELLOW_RED')
#cmap=plt.get_cmap('autumn')
#cmap=plt.get_cmap('gist_heat')
#cmap=plt.get_cmap('Spectral_r')
#cmap.set_under([0.0, 0.0, 0.0])



# Load profile
profileName = os.path.abspath("foam_ball_short.dsa")
frameID = 230

frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileName);
numTSFrames = frameManager.get_tsframe_count();

matrixID = 1

# Load single frame
tsframe = np.copy( frameManager.get_tsframe(frameID, matrixID) );

cols = tsframe.shape[1]
rows = tsframe.shape[0]

# Add padding on border
padding = 2
v_padding = np.empty((padding, cols)); v_padding.fill(-50)
h_padding = np.empty((rows+2*padding, padding)); h_padding.fill(-50)
zs = np.vstack([v_padding, tsframe]) # Top
zs = np.vstack([zs, v_padding]) # Bottom
zs = np.hstack([h_padding, zs]) # Left
zs = np.hstack([zs, h_padding]) # Right

# Update matrix size with padding
cols = zs.shape[1]
rows = zs.shape[0]

# Coordinates of sampled data points
xs = np.arange(0, cols, 1)
ys = np.arange(0, rows, 1)

# Coordinates of interpolation points
scaleFactor = 10;
xi = np.linspace(xs.min(), xs.max(), cols*scaleFactor)
yi = np.linspace(ys.min(), ys.max(), rows*scaleFactor)





#------------------------------------------------------
# Interpolate with cubic splines
spline = scipy.interpolate.RectBivariateSpline(ys, xs, zs, kx=3, ky=3, s=0)

# Evaluate splines
zi = spline(yi, xi)

#------------------------------------------------------


'''
#------------------------------------------------------
# Polynomial interpolation: ‘linear’, ‘nearest’, ‘cubic’
coordinates = [(y, x) for y in ys for x in xs]
zs_flattened = np.ravel(zs, order='C')
coordinates_interpolated = [(y, x) for y in yi for x in xi]

# Interpolate with griddata
zi_flattened= scipy.interpolate.griddata(coordinates, zs_flattened, coordinates_interpolated, method='cubic')

# Reshape flattened array to 2D
zi = zi_flattened.reshape((rows*scaleFactor, cols*scaleFactor))
#------------------------------------------------------
'''




#------------------------------------------------------
# Old API
# Set up a regular grid of sampled data points
#ys, xs = np.meshgrid(xs, ys)

# Set up a regular grid of interpolated points
#yi, xi = np.meshgrid(xi, yi)

# Interpolate
#tck = scipy.interpolate.bisplrep(xs2, ys2, zs, kx=3, ky=3, s=0)

# Evaluate splines
#zi = scipy.interpolate.bisplev(xi2[:,0], yi2[0,:], tck)
#------------------------------------------------------




# Apply threshold to level out small values (interpolation ripples)
min_threshold = 25
zi[zi < min_threshold ] = 0 



#########################################
# Plotting
#########################################
fig, ax = plt.subplots()


############
# Histogram
############
plt.hist(zi.flatten(), 128, range=(min_threshold, zi.max()), fc='k', ec='k')
plt.savefig("histogram.pdf", format='pdf')
plt.close() 

########################
# Interpolated image
########################
fig, ax = plt.subplots()

# Interpolated image
#cmap=plt.get_cmap('gray')
cmap=plt.get_cmap('bds_highcontrast')
cax = ax.imshow(zi, cmap=cmap, vmin=zs.min(), vmax=zs.max(), origin='lower', extent=[xs.min(), xs.max(), ys.min(), ys.max()])

# Colorbar with countour levels
cbar = fig.colorbar(cax)
cbar.set_label('Raw sensor value', rotation=90)
cbar.solids.set_edgecolor("face") # set the color of the lines

ax.invert_yaxis()
ax.xaxis.tick_top()
plt.axis('off')

plt.savefig("interpolation.pdf", format='pdf')
plt.close() 



############
# Contour
############
fig, ax = plt.subplots()

# Nearest-Neighbor Image
cax = ax.imshow(zs, interpolation='nearest', cmap=plt.get_cmap('gray'), vmin=zs.min(), vmax=zs.max(), origin='lower', extent=[xs.min(), xs.max(), ys.min(), ys.max()]) 

#------------------------------------------------------
# Contour lines: contour()
#------------------------------------------------------
countour_threshold = 50
levels = np.linspace(countour_threshold, zs.max(), 10)
#contour = ax.contour(xi, yi, zi, levels, linewidths=1.0, colors=[(0.0, 0.0, 0.0)], origin='upper') # black contour
contour = ax.contour(xi, yi, zi, levels, linewidths=1.0, colors=[(1.0, 0.0, 0.0)], origin='upper') # Red contour
#contour = ax.contour(xi, yi, zi, levels, linewidths=1.0, cmap=plt.get_cmap('bds_highcontrast'), origin='upper') # Colormap

#plt.clabel(contour, inline=True, fontsize=9)

# Colorbar with countour levels
cbar = fig.colorbar(cax)
cbar.add_lines(contour)
cbar.set_label('Raw sensor value', rotation=90)
cbar.solids.set_edgecolor("face") # set the color of the lines

'''
#------------------------------------------------------
# Filled contours: contourf()
#------------------------------------------------------

# Background image
background = np.empty((rows, cols)); background.fill(0)
cax = ax.imshow(background, cmap=plt.get_cmap('gray'), origin='lower', extent=[xs.min(), xs.max(), ys.min(), ys.max()] )

# Filled contour
countour_threshold = 100 # Ignore "ripples" from spline extrapolation
max_threshold = 0 # Boost the upper limit to avoid truncation error
levels = np.linspace(countour_threshold, zs.max(), num=10, endpoint=True)

# Levels correspond to midpoint of layers:
# Extend level range to enlarge top layer (avoid ugly hole)
levels[-1] = levels[-1] + (levels[-1] - levels[-2])/2

contour = ax.contourf(xi, yi, zi, levels=levels, cmap=plt.get_cmap('bds_highcontrast'), origin='upper')      

cbar = fig.colorbar(contour, format='%.0f')
cbar.set_label('mV', rotation=0)
cbar.solids.set_edgecolor("face") # set the color of the lines

# Restore old levels
#levels[-1] = zs.max() 
#cbar.set_ticks(levels)
#------------------------------------------------------
'''


ax.invert_yaxis()
ax.xaxis.tick_top()
plt.axis('off')

plt.savefig("contour.pdf", format='pdf')
plt.show() 



 
