import numpy as np
import matplotlib.pyplot as plt

import cv2
import struct

# Convert png to dsa (for visualization purposes only)

tsframe0 = cv2.imread('matrix0.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
tsframe1 = cv2.imread('matrix1.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
tsframe2 = cv2.imread('matrix2.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
tsframe3 = cv2.imread('matrix3.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
tsframe4 = cv2.imread('matrix4.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
tsframe5 = cv2.imread('matrix5.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)

# Normalize frames
max_val = 3000.0
tsframe0 = tsframe0.astype(np.float32) * (max_val/255.0)
tsframe1 = tsframe1.astype(np.float32) * (max_val/255.0)
tsframe2 = tsframe2.astype(np.float32) * (max_val/255.0)
tsframe3 = tsframe3.astype(np.float32) * (max_val/255.0)
tsframe4 = tsframe4.astype(np.float32) * (max_val/255.0)
tsframe5 = tsframe5.astype(np.float32) * (max_val/255.0)

# Ensure masked taxels are 0
masked_idx = np.array([[0,0], [1,0], [2,0], [3,0], [4,0], [0,5], [1,5], [2,5], [3,5], [4,5]]).T
tsframe1[masked_idx[0], masked_idx[1]] = 0.0
tsframe3[masked_idx[0], masked_idx[1]] = 0.0
tsframe5[masked_idx[0], masked_idx[1]] = 0.0


############
# Plotting
############
def show_matrix(tsframe):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #colormap=plt.get_cmap('gray')
    colormap=plt.get_cmap('YlOrRd_r')
    #colormap = plt.get_cmap('afmhot')
    #colormap.set_under([0.0, 0.0, 0.0])
    colormap.set_under([0.2, 0.2, 0.2])
    #colormap.set_under([1.0, 1.0, 1.0])

    width = tsframe.shape[1]; height = tsframe.shape[0]
    xs,ys = np.meshgrid(np.arange(0, width+1), np.arange(0, height+1))
    # Workaround inverted y-axis
    ax.invert_yaxis()

    # pcolormesh aligns cells on their edges, while imshow aligns them on their centers.
    ax.pcolormesh(xs-0.5, ys-0.5, tsframe, cmap=colormap, vmin=0.1, vmax=tsframe.max(),
                  shading="faceted", linestyle="dashed", linewidth=0.5, edgecolor=[0.0, 0.0, 0.0])

    # Absolute number
    for i,j in ((x,y) for x in np.arange(0, len(tsframe))
        for y in np.arange(0, len(tsframe[0]))):
            if tsframe[i][j] >= 1:
                ax.annotate(str(int(tsframe[i][j])), xy=(j,i), fontsize=8, ha='center', va='center')

    ax.set_aspect('equal')
    ax.set_xlim([-0.5, width-0.5])
    ax.set_ylim([height-0.5, -0.5])

    ax.xaxis.tick_top()
    ax.tick_params(axis='both', which='both', left='off', right='off', bottom='off', top='off', labeltop='on')
    #ax.xaxis.set_major_locator(plt.NullLocator())
    #ax.yaxis.set_major_locator(plt.NullLocator())
    plt.show()
       
#show_matrix(tsframe0)


# Flatten matrices according to taxel indices
tsframe = np.concatenate([np.reshape(tsframe0, -1, order='C'),
                        np.reshape(tsframe1, -1, order='C'),
                        np.reshape(tsframe2, -1, order='C'),
                        np.reshape(tsframe3, -1, order='C'),
                        np.reshape(tsframe4, -1, order='C'),
                        np.reshape(tsframe5, -1, order='C')])


def tohex(value):
    buf = struct.pack('<f', value) # this is like the C cast to uint32_t
    val = struct.unpack('I', buf)[0] 
    return "{0:0{1}X}".format(val,8)


# Frames are encoded using an "Enhanced RLE Compression"
# Meaning only zero-valued matrix elements are run length encoded.
# Each token t consists of a single precision floating point value (IEEE-754)
# t < 0 indicates |t| consecutive zeros
# t > 0: The value of t represents a single observation
consecutiveZeros = 0.0;
hexdata = "" 
for value in tsframe:
    if(value == 0.0):
        consecutiveZeros += 1.0;
    else:
        if(consecutiveZeros > 0):
            hexdata += tohex(-consecutiveZeros) # hex representation of negative number of consecutive zeros
            consecutiveZeros = 0.0
        hexdata += tohex(value) # hex representation of cell value
        
if(consecutiveZeros > 0.0): # Don't forget trailing zeros
	hexdata += tohex(-consecutiveZeros)



print hexdata

