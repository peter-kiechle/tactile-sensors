# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Library path
print("CWD: " + os.getcwd() )
lib_path = os.path.abspath('../../lib')
sys.path.append(lib_path)
import framemanager_python

# Load profile
profileName = os.path.abspath("teapot_handle.dsa")
frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileName);
frameManager.set_filter_median(1, True)

numTSFrames = frameManager.get_tsframe_count();
starttime = frameManager.get_tsframe_timestamp(0)
stoptime = frameManager.get_tsframe_timestamp(numTSFrames)

frameID = 230
matrixID = 1

tsframe = np.copy( frameManager.get_tsframe(frameID, matrixID) );


# Alternative: Draw from PNG  
#tsframe = cv2.imread("img_rectangle_v_reconstructed.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)


############
# Plotting
############
def show_matrix(tsframe):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colormap=plt.get_cmap('YlOrRd_r')
    colormap.set_under([0.2, 0.2, 0.2])

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
    plotname = "plot_matrix"
    fig.savefig(plotname+".pdf", pad_inches=0, bbox_inches='tight', dpi=fig.dpi) # pdf
    #fig.savefig(plotname+".pgf", pad_inches=0, bbox_inches='tight', dpi=fig.dpi) # pgf
    
show_matrix(tsframe)
 