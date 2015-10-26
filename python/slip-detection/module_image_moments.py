# -*- coding: utf-8 -*-

import numpy as np
import cv2
from math import atan2, pi

#from scipy import ndimage
  
def compute_orientation_and_shape_features(image):

    image64 = np.float64(image)
  
  
    # Upscaling
    # Order
    # 0: Nearest-neighbor
    # 1: Bi-linear (default)
    # 2: Bi-quadratic
    # 3: Bi-cubic
    # 4: Bi-quartic
    # 5: Bi-quintic
    #image64 = ndimage.zoom(image64, 4, order=2)
  
  
  
    # Normalize image such that M['m00'] = 1
    # Only necessary for computation of standard deviation and skew
    image_normalized = image64/np.sum(image64) 
     
    # Compute geometric moments of greyscale image
    M = cv2.moments(image_normalized, binaryImage=False) 
   
    # Make formulas look nicer
    # Raw moments
    m00 = M['m00']
    m01 = M['m01'] 
    m10 = M['m10']
    m02 = M['m02']
    m20 = M['m20']
    
    # Central moments
    mu00 = m00
    mu11 = M['mu11']
    mu02 = M['mu02']
    mu20 = M['mu20']
    mu03 = M['mu03']
    mu30 = M['mu30']

    # Normalized moments
    mu11_ = mu11/mu00
    mu02_ = mu02/mu00
    mu20_ = mu20/mu00
    
    #mu11_ = mu11
    #mu02_ = mu02
    #mu20_ = mu20
    
        
    #-------------------------------------------------------------------    
    # Covariance matrix
    #-------------------------------------------------------------------
    Cov = np.array([[mu20_, mu11_], [mu11_, mu02_]])

    
    #-------------------------------------------------------------------
    # Eigenvalues
    #-------------------------------------------------------------------
    lambda1 = 0.5*(mu20_ + mu02_) + 0.5*(mu20_**2 + mu02_**2 - 2*mu20_*mu02_ + 4*mu11_**2)**0.5
    lambda2 = 0.5*(mu20_ + mu02_) - 0.5*(mu20_**2 + mu02_**2 - 2*mu20_*mu02_ + 4*mu11_**2)**0.5
     
    #------------------------------------------------------------------- 
    # Eigenvectors
    #-------------------------------------------------------------------
    #u = np.array([ -(mu11_*mu11_) / (lambda1*mu02_*(-mu20_+lambda1)), -mu11_/(mu02_+lambda1)])
    #v = np.array([ -(mu11_*mu11_) / (lambda2*mu02_*(-mu20_+lambda2)), -mu11_/(mu02_+lambda2)])
     
     
    #-------------------------------------------------------------------
    # Orientation angle
    #-------------------------------------------------------------------   
    # ±90 degrees
    angle_rad = (0.5*atan2((2*mu11), (mu20-mu02)))
    angle_deg = angle_rad * 180.0/pi;  

    # Alternative angle computation (ambiguity: 90 degrees)
    #angle = atan((lambda_max - mu20)/mu11) * 180.0/pi

    # clamp to [0, 180]
    if (angle_deg < 0):
        angle_deg = 90 + (90+angle_deg)


    #-------------------------------------------------------------------
    # Geometric moment based statistics
    # From Moments and Moment Invariants in Pattern Recognition:
    # In case of zero means, m20 and m02 are variances of horizontal 
    # and vertical projections and m11 is a covariance between them
    #-------------------------------------------------------------------
    if( m00  > 0.0):    
        centroid_x = m10 / m00
        centroid_y = m01 / m00
    else: 
        raise Exception, 'Empty frame!'

    var_x = m20 - centroid_x * m10
    var_y = m02 - centroid_y * m01
    std_dev_x = np.sqrt(var_x)
    std_dev_y = np.sqrt(var_y)

    if abs(mu20) > 0.001:
        skew_x = mu30 / np.power(mu20, 1.5)
    else:
        skew_x = 0.0
        
    if abs(mu02) > 0.001:
        skew_y = mu03 / np.power(mu02, 1.5)
    else:
        skew_y = 0.0
    
    
    #-------------------------------------------------------------------
    # Compute compactness measure based on moments
    #-------------------------------------------------------------------
    # M['m00'] == cv2.countNonZero(img) in case of binaryImage=True    
    area = cv2.countNonZero(image_normalized)
    compactness1 = area / (2.0*np.pi * (mu20 + mu02) )

    
    #-------------------------------------------------------------------
    # Compute compactness measure based on contour area and perimeter
    #-------------------------------------------------------------------
    # Add border padding due to strange behavior of cv::findContours() in case of contours touching the image border
    image_padded = cv2.copyMakeBorder(image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 0, 0) )
    #image_padded = np.uint8(image_padded * 255.0) # scale to [0..255] and convert to uint8
    image_padded = cv2.normalize(image_padded, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.cv.CV_8UC1); # Scale such that highest intensity is 255
    contours, hierarchy = cv2.findContours(image_padded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #For now, just take the largest perimeter and area
    perimeter = 0
    area_contour = 0
    for i in range(len(contours)):
        cnt = contours[i]
        perimeter = max( perimeter, cv2.arcLength(cnt, True) )
        area_contour = max( area_contour, cv2.contourArea(cnt) )

    compactness2 = (4*np.pi*area_contour) / perimeter**2

    # Show contour
    #cv2.drawContours(image_padded, contours,-1, (255,255,255), 1)
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow('image', image_padded)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    #-------------------------------------------------------------------
    # Compute compactness measure based on minimal enclosing circle
    #-------------------------------------------------------------------
    #(x,y),radius = cv2.minEnclosingCircle(cnt)
    #area_circle = radius**2 * np.pi
    #compactness3 = area_contour / area_circle
   
    
    #-------------------------------------------------------------------
    # Compute eccentricity (elongation)
    #-------------------------------------------------------------------
    eccentricity1 = np.sqrt(1.0 - lambda2 / lambda1);
    eccentricity2 = np.sqrt(1.0 - (lambda2 / lambda1)**2)    
    
    
    return (centroid_x, centroid_y, angle_deg, Cov, lambda1, lambda2, 
            std_dev_x, std_dev_y, skew_x, skew_y,
            compactness1, compactness2, eccentricity1, eccentricity2)
    

def report_shape_features(description, features):
    threshold_skewness = 0.08
 
    (centroid_x, centroid_y, angle_deg, Cov, lambda1, lambda2, 
     std_dev_x, std_dev_y, skew_x, skew_y,
     compactness1, compactness2, eccentricity1, eccentricity2) = features
    
    # Regrouping
    features =  (centroid_x, centroid_y, std_dev_x, std_dev_y, skew_x, skew_y,
                 angle_deg, lambda1, lambda2,
                 compactness1, compactness2, eccentricity1, eccentricity2)

    names = ("Centroid x", "Centroid y", "Standard deviation x", "Standard deviation y", "Skewness x", "Skewness y",
             "Angle", "Major axis", "Minor axis", 
             "Compactness (moments)", "Compactness (contour)", "Eccentricity", "Eccentricity (squared)")

    print "\n----------------------------------\nFeatures: ", description, "\n----------------------------------"
   
    for name, val in zip(names, features):

        if name == "Skewness x":
            if val < -threshold_skewness:
                print "%-25s % .4f  Mass is concentrated on the right" %(name, val)
            elif val > threshold_skewness :
                print "%-25s % .4f  Mass is concentrated on the left" %(name, val)
            else:
                print "%-25s % .4f  Mass is horizontally centered" %(name, val)

        elif name == "Skewness y":       
            if val < -threshold_skewness:
                print "%-25s % .4f  Mass is concentrated in the lower region" %(name, val)
            elif val > threshold_skewness:
                print "%-25s % .4f  Mass is concentrated in the upper region" %(name, val)
            else:
                print "%-25s % .4f  Mass is vertically centered" %(name, val)
            
        else:
            print '%-25s % .4f' %(name, val)


def valid_frame(compactness, eccentricity, thresh_compactness, thresh_eccentricity):
    valid = False    
    if eccentricity > thresh_eccentricity: # Elongated shape
        valid = True;
    else:
        if compactness < thresh_compactness: # At least not circular
            valid = True;
        
    return valid  


def get_quadrant(angle):
    """Determines the quadrant angle is in"""
    
    r = abs(angle) % 360.0
    if(angle < 0): # Attention: Python's modulo differs from c's
        r = -r
    angle = r
    
    if ((angle >= 0.0 and angle < 90.0) or (angle >= -360.0 and angle < -270.0)):
        return 1
    elif ((angle >= 90.0 and angle < 180.0) or (angle >= -270.0 and angle < -180.0)):
        return 2
    elif ((angle >= 180.0 and angle < 270.0) or (angle >= -180.0 and angle < -90.0)):
        return 3
    else:
        return 4
       
def angle_abs_difference(angle0, angle1):
    return abs((angle0 + 180 -  angle1) % 360 - 180)

def angle_difference(angle0, angle1):
    difference = angle1 - angle0;
    while (difference < -180): difference += 360;
    while (difference > 180): difference -= 360;
    return difference;


def track_angle(reference_angle, previous_angle, angle, n):
    current_angle1 = angle
    current_angle2 = 180.0+angle    
    
    angle_diff1 = angle_difference(previous_angle, current_angle1)
    angle_diff2 = angle_difference(previous_angle, current_angle2)

    if( abs(angle_diff1) < abs(angle_diff2) ):
        current_angle = current_angle1
        angle_diff = angle_diff1
    else:
        current_angle = current_angle2
        angle_diff = angle_diff2
        
        
    # Discontinuiuty 360 -> 0
    if (current_angle < previous_angle and angle_diff > 0):
        #print("discontinuiuty IV -> I")
        n += 1
   
    # Discontinuiuty 0 -> 360
    if (current_angle > previous_angle and angle_diff < 0):
        #print("discontinuiuty I -> IV")
        n -= 1    
    
    # Absolute slip angle (to reference frame)
    slip_angle_reference = n*360.0 + current_angle + angle_difference(reference_angle, 0.0)

    # Differential slip angle (to previous frame)
    slip_angle = angle_diff

    #print( "Reference: % 7.2f,  Current: % 7.2f, Slip: % 7.2f (% 7.2f)"%(reference_angle, current_angle, slip_angle, slip_angle_reference) )
    return current_angle, slip_angle, slip_angle_reference, n



