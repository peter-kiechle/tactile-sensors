#!/usr/bin/python
# Computes the Root Mean Square Error between two images
from sys import argv
import numpy as np
import math
import cv2

#img1 = str(argv[1])
#img2 = str(argv[2])

infile1 = "img_doughnut_padded.png"
infile2 = "img_doughnut_reconstructed_5.png"

img1 = cv2.imread(infile1, cv2.CV_LOAD_IMAGE_GRAYSCALE)
img2 = cv2.imread(infile2, cv2.CV_LOAD_IMAGE_GRAYSCALE)

img1 = np.array(img1, dtype=np.float32)
img2 = np.array(img2, dtype=np.float32)

error = np.absolute(img1 - img2)
rmse_np = np.mean((error**2).flatten())**0.5
print "RMSE: ", rmse_np, rmse_np / 256