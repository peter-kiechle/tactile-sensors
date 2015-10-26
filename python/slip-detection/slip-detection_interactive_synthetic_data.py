#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from math import pi
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import module_image_moments as IM
import module_normalized_cross_correlation as NCC

# Reloading not needed in final version
reload(IM)
reload(NCC)


UIBK_blue = [0.0, 0.1765, 0.4392]
UIBK_orange = [1.0, 0.5, 0.0]


def generate_frame(angle, shift) :
        
    symmetric=False
    scale = 3
    shift *= scale
    width=6*scale; height=14*scale; 
    
    
    if symmetric is False:  # Create rectancle
      
        padding = 10 # padding is neccessary due to a bug in cv2.line and will be removed afterwards
        # Create image with 
        image = np.zeros((height+2*padding, width+2*padding), np.uint8)
        rotation_center = tuple([width/2.0+padding, padding+height/2 + shift+1])
        radius = 100

        p1 = np.array([int(round(rotation_center[0] + (radius * np.cos(np.radians(angle))))), 
                       int(round(rotation_center[1] + (radius * np.sin(np.radians(angle))))) ])

        p2 = np.array([int(round(rotation_center[0] + (radius * np.cos((angle+180)%360 * pi / 180.0)))), 
                       int(round(rotation_center[1] + (radius * np.sin((angle+180)%360 * pi / 180.0))))])

        cv2.line(image, tuple(p1), tuple(p2), (196,0,0), 5, cv2.CV_AA)
        cv2.line(image, tuple(p1), tuple(p2), (255,0,0), 1, cv2.CV_AA)    
    
        image = image[padding:padding+height, padding:padding+width]  # Trim padding

    else: # Create square
       
        r_w = 2 # width
        r_h = 2 # height
             
        image = np.zeros((height, width), np.uint8)
        rotation_center = tuple([width/2, height/2]) # x,y
         
        pt1 = tuple([(int)(rotation_center[0]-r_w*scale), (int)(rotation_center[1]-r_h*scale)])
        pt2 = tuple([(int)(rotation_center[0]+r_w*scale), (int)(rotation_center[1]+r_h*scale)])
        cv2.rectangle(image, pt1, pt2, 128, thickness=cv2.cv.CV_FILLED ) #cv2.FILLED
    
        #Rotation
        rot_mat = cv2.getRotationMatrix2D(rotation_center, angle, 1.0)
        image = cv2.warpAffine(image, rot_mat, (width,height), flags=cv2.INTER_LINEAR)
    
        # Translation
        trans_mat = np.float32([[1, 0, 0],[0, 1, shift]])
        image = cv2.warpAffine(image, trans_mat, (width,height))
   
    if (scale > 1):
        image = cv2.resize(image, None, fx=1.0/scale, fy=1.0/scale, interpolation = cv2.INTER_CUBIC)
        image = cv2.threshold(image, 0.0, 1.0, cv2.THRESH_TOZERO)[1] # Remove interpolation artefacts
    
    image = np.float32(image/255.0) # scale to [0..1] and convert to float
    return image
    




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
    ax.axis('image')
    return artists_minor, artists_major
   

def update_frame(val):
    
    global n     
    global previous_angle    
    
    angle = np.around(slider_rotation.val)
    shift = np.around(slider_translation.val)
    
    frame1 = generate_frame(angle, shift) # Slip simulation
    
    # Compute translation
    slipvector = NCC.normalized_cross_correlation(frame0, frame1)
        
    # Compute orientation
    (centroid_x, centroid_y, angle, Cov, lambda1, lambda2, 
     std_dev_x, std_dev_y, skew_x, skew_y,
     compactness1, compactness2, eccentricity1, eccentricity2) = IM.compute_orientation_and_shape_features(frame1)
    
    # Track rotation   
    current_angle, slip_angle, slip_angle_reference, n = IM.track_angle(reference_angle, previous_angle, angle, n)
 
    previous_angle = current_angle

    ###########
    # Plotting
    ###########
    ax = axes[1]
    ax.cla() # clear
    ax.imshow(frame1, cmap=cmap, vmin=0.001, vmax=1.0, interpolation='nearest')
    l3, l4 = plot_principal_axes_cov(centroid_x, centroid_y, Cov, UIBK_orange, ax)
    ax.set_title("Moving frame")
    ax.text(0.01, 1.0, text_template_angle%(current_angle, slip_angle), size=12, color=UIBK_orange)
    ax.text(0.01, 2.0, text_template_shift%(slipvector[0], slipvector[1]), size=12, color=UIBK_orange)
    plt.draw()







angle_start = 0 # [0, 180)
shift_start = 0

# Generate initial frame
frame0 = generate_frame(angle_start, shift_start) # Slip simulation

# Compute initial orientation
(centroid_x, centroid_y, angle, Cov, lambda1, lambda2, 
 std_dev_x, std_dev_y, skew_x, skew_y,
 compactness1, compactness2, eccentricity1, eccentricity2) = IM.compute_orientation_and_shape_features(frame0)

reference_angle = angle # [0, 180)
previous_angle = angle # [0, 360)
slip_angle = 0 # (-∞, ∞)
n = 0 # rotation carry



###########
# Plotting
###########

#mpl.rcParams['text.usetex'] = True  # True, False
text_template_angle = r'$\mathbf{ \phi: %.1f \quad \Delta \phi = %.1f} $'
text_template_shift = r'$\mathbf{ \Delta x: %.1f, \quad \Delta y: %.1f} $'
#text_template_angle = 'phi: %.1f   dphi = %.1f'
#text_template_shift = 'dx: %.1f,   dy: %.1f'

cmap=plt.get_cmap('gray')
cmap.set_under([0.0, 0.0, 0.0])


fig_anim, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, squeeze=True, dpi=100)
ax = axes[0]
ax.imshow(frame0, cmap=cmap, vmin=0.001, vmax=1.0, interpolation='nearest')
l1, l2 = plot_principal_axes_cov(centroid_x, centroid_y, Cov, UIBK_orange, ax)
ax.set_title("Reference frame")

plt.subplots_adjust(top=0.88, left = 0.07, bottom=0.2, right = 1.0)

ax_slider_translation = plt.axes([0.25, 0.10, 0.6, 0.03]) # left, bottom, width, height
ax_slider_rotation = plt.axes([0.25, 0.05, 0.6, 0.03]) # left, bottom, width, height

slider_translation = Slider(ax_slider_translation, 'Translation', -7, 7, valfmt='%0.0f', valinit=shift_start)
slider_rotation = Slider(ax_slider_rotation, 'Rotation', -540, 540, valfmt='%0.0f', valinit=angle_start)

slider_translation.on_changed(update_frame)
slider_rotation.on_changed(update_frame)

update_frame(42)
plt.show()

