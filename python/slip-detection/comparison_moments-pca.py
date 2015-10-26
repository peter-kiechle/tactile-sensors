# -*- coding: utf-8 -*-

import os, sys
config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)
import configuration as config

import cv2
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

import module_image_moments as IM
reload(IM) # Not needed in final version

'''
def plot_principal_axes_angle(c_x, c_y, major_axis_width, minor_axis_width, angle, color, ax):
   """Plot principal axes using specified angle and length
      Taken from Joe Kington: http://stackoverflow.com/questions/9005659/compute-eigenvectors-of-image-in-python
   """
   def plot_bar(r, c_x, y_bar, angle, ax, color):
        dx = r * np.cos(np.radians(angle))
        dy = r * np.sin(np.radians(angle))
        ax.plot([c_x - dx, c_x, c_x + dx], 
                [c_y - dy, c_y, c_y + dy], '-', color=color, linewidth=3.0, alpha=0.9)
        """
        ax.annotate("",
                    xy=(c_x-dx, c_y-dy), xycoords='data',
                    xytext=(c_x+dx, c_y+dy), textcoords='data',
                    arrowprops=dict(arrowstyle="|-|", color = color, linewidth=3.0)
                    )        
        """
   plot_bar(minor_axis_width, c_x, c_y, angle+90.0, ax, color=color) # Minor axis
   plot_bar(major_axis_width, c_x, c_y, angle, ax, color=color) # Major axis
   #ax.plot(c_x, c_y, 'o', color=color, linewidth=4.0) # Center
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
    ax.plot(*make_lines(eigvals, eigvecs, mean, 0), color=color, linewidth=3.0, alpha=0.9)
    ax.plot(*make_lines(eigvals, eigvecs, mean, -1), color=color, linewidth=3.0, alpha=0.9)
    ax.axis('image')
'''  





def plot_principal_axes_angle(c_x, c_y, major_axis_width, minor_axis_width, angle, color, ax):
   """Plot principal axes using specified angle and length
      Taken from Joe Kington: http://stackoverflow.com/questions/9005659/compute-eigenvectors-of-image-in-python
   """
   def plot_bar(r, c_x, y_bar, angle, ax, color):
        dx = r * np.cos(np.radians(angle))
        dy = r * np.sin(np.radians(angle))
        ax.plot([c_x - dx, c_x, c_x + dx], 
                [c_y - dy, c_y, c_y + dy], '-', color=color, linewidth=3.0, alpha=1.0)
        """
        ax.annotate("",
                    xy=(c_x-dx, c_y-dy), xycoords='data',
                    xytext=(c_x+dx, c_y+dy), textcoords='data',
                    arrowprops=dict(arrowstyle="|-|", color = color, linewidth=3.0)
                    )        
        """
   plot_bar(minor_axis_width, c_x, c_y, angle+90.0, ax, color=color) # Minor axis
   plot_bar(major_axis_width, c_x, c_y, angle, ax, color=color) # Major axis
   #ax.plot(c_x, c_y, 'o', color=color, linewidth=4.0) # Center
   #ellipse = mpl.patches.Ellipse(xy=(c_x, c_y), width=2*major_axis_width, height=2*minor_axis_width, angle=angle, 
   #                              facecolor=[0.0, 0.0, 0.0, 0.0], edgecolor=color, linestyle='solid', linewidth=2.0 )
   #ax.add_artist(ellipse)
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
    ax.plot(*make_lines(eigvals, eigvecs, mean, 0), color=color, linewidth=3.0, alpha=0.9)
    ax.plot(*make_lines(eigvals, eigvecs, mean, -1), color=color, linewidth=3.0, alpha=0.9)
    ax.axis('image')



#image = cv2.imread('blob_symmetric.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
image = cv2.imread('blob_asymmetric.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)



#################
# Image Moments
################

(centroid_x, centroid_y, angle_deg, Cov, lambda1, lambda2, 
 std_dev_x, std_dev_y, skew_x, skew_y,
 compactness1, compactness2, eccentricity1, eccentricity2) = IM.compute_orientation_and_shape_features(image)
 

#########
## PCA
#########
# Extract image coordinates of active cells
active_cells = np.array(np.where(image > 0.001)).T
active_cells = np.fliplr(active_cells) # x in first column, y in second

# Center the data
center = np.mean(active_cells, axis=0)
active_cells -= center

# Calculate the covariance matrix
Cov_PCA = np.cov(active_cells.T)


############
# Plotting
############
colormap=plt.get_cmap('gray')
#colormap=plt.get_cmap('YlOrRd')
#colormap = plt.get_cmap('afmhot')
colormap.set_under([0.0, 0.0, 0.0])
#colormap.set_under([0.3, 0.3, 0.3])
#colormap.set_under([1.0, 1.0, 1.0])

text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0

size_factor = 0.75
figure_width = size_factor*text_width
figure_height = (figure_width / golden_ratio)
#figure_height = 1.3 * figure_width
figure_size = [figure_width, figure_height]

config.load_config_medium()





fig = plt.figure(figsize=figure_size)
ax = plt.subplot()


# Workaround inverted y-axis
ax.invert_yaxis()
#image = np.flipud(image)

width = image.shape[1]
height = image.shape[0]
xs,ys = np.meshgrid(np.arange(0, width+1), np.arange(0, height+1))

# pcolormesh aligns cells on their edges, while imshow aligns them on their centers.
ax.pcolormesh(xs-0.5, ys-0.5, image, cmap=colormap, vmin=0.001, vmax=255.0,
              shading="faceted", linestyle="dashed", linewidth=1.0, edgecolor=[0.0, 0.0, 0.0])

ax.set_aspect('equal')
ax.xaxis.tick_top()
plt.tick_params(axis='both', which='both', left='off', right='off', bottom='off', top='off',  labeltop='on')


# PCA
plot_principal_axes_cov(center[0], center[1], Cov_PCA, config.UIBK_blue, ax)

# Second order moments
plot_principal_axes_angle(centroid_x, centroid_y, lambda1, lambda2, angle_deg, config.UIBK_orange, ax)
#plot_principal_axes_cov(c_x, c_y, Cov_moments, UIBK_orange, ax)


# Legend
dummy_moments, = plt.plot((0, 0), (1, 1), '-', color=config.UIBK_orange, alpha=1.0)
dummy_PCA, = plt.plot((0, 0), (1, 1), '-', color=config.UIBK_blue, alpha=1.0)
ax.legend([dummy_moments, dummy_PCA], [r"Second order moments", r"Principal Component Analysis"], 
              loc = 'upper center', bbox_to_anchor=(0.5, -0.025), fancybox=True, shadow=False )

#loc = 'center left', bbox_to_anchor=(1.2, 0.5)
#loc = 'upper center', bbox_to_anchor=(0.5, 1.23),

fig.tight_layout()
fig.show()

plotname = "comparison_moments-pca"
fig.savefig(plotname+".pdf", pad_inches=0, bbox_inches='tight', dpi=fig.dpi) # pdf
#fig.savefig(plotname+".pgf", pad_inches=0, bbox_inches='tight', dpi=fig.dpi) # pgf

