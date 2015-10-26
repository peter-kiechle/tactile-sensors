# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import DenavitHartenberg as DH
import mpl_toolkits.mplot3d as Axes3d # @UnusedImport

from scipy.spatial import ConvexHull
import cv2

# Load configuration file before pyplot
import os, sys
config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)
import configuration as config

# Library path
import os, sys
print("CWD: " + os.getcwd() )
lib_path = os.path.abspath('../../lib')
sys.path.append(lib_path)

import framemanager_python
# Force reloading of libraries (convenient during active development)
reload(DH)
reload(framemanager_python)


UIBK_blue = [0.0, 0.1765, 0.4392]
UIBK_orange = [1.0, 0.5, 0.0]

def load_image(filename):
    img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE).astype(np.float32)
    try:
        img.data
    except:
       print "Error loading file"

    img /= 255.0 # [0,1]
    return img


def project_active_cells_proximal(tsframe, T02):
    points = np.empty([0,3])
    for y in range(14):
        for x in range(6):
            if(tsframe[y,x] > 0.0):
                vertex = DH.get_xyz_proximal(x, y, T02)
                points = np.vstack((points, vertex))
    return points       



def project_active_cells_distal(tsframe, T03):
    points = np.empty([0,3])
    for y in range(13):
        for x in range(6):
            if(tsframe[y,x] > 0.0):
                vertex = DH.get_xyz_distal(x, y, T03)
                points = np.vstack((points, vertex))
    return points   



def compute_diameter(points):
    
    # Compute convex hull (using qhull)
    hull = ConvexHull(points)

    # Compute diameter - Brute force, O(n^2), see "Rotating calipers algorithm" for O(n) solution
    diameter = 0.0
    endpoints_idx = np.array([0, 1]) # Index of diameter endpoints in convex hull
    for v_i in hull.vertices:
        for v_j in  hull.vertices:
            diff = points[v_i] - points[v_j]
            distance = np.sqrt(diff.dot(diff))
            if(distance > diameter):
                diameter = distance
                endpoints_idx[0] = v_i
                endpoints_idx[1] = v_j
    
    return diameter, hull, endpoints_idx
    


def plot_convex_hull(ax, points, hull, endpoints_idx):
    # 3D convex hull
    ax.plot_trisurf(points[:,0], points[:,1], points[:,2], 
                    triangles=hull.simplices, shade=True, antialiased=True,
                    edgecolor=[1.0, 0.5, 0.0, 0.4], linewidth=0.0,
                    color=[1.0, 0.5, 0.0, 0.25])
    # Points
    ax.plot(points[:,0], points[:,1], points[:,2], 'ro')
    
    # Diameter
    #ax.plot(points[endpoints_idx,0], points[endpoints_idx,1], points[endpoints_idx,2], '--', color=UIBK_blue, lw=2) 



def plot_cells_proximal(ax, tsframe, T02, scalarMap):
    span = 0.5
    for y in range(14):
        for x in range(6):
            vertex = np.empty([4,3])
            vertex[0,:] = DH.get_xyz_proximal(x-span, y+span, T02)
            vertex[1,:] = DH.get_xyz_proximal(x+span, y+span, T02)
            vertex[2,:] = DH.get_xyz_proximal(x+span, y-span, T02)
            vertex[3,:] = DH.get_xyz_proximal(x-span, y-span, T02)
            quad = Axes3d.art3d.Poly3DCollection([vertex])
            
            color = scalarMap.to_rgba(tsframe[y,x])            
                        
            if(tsframe[y,x] > 0.0):
                quad.set_color([color[0], color[1], color[2], 1.0]) # Active cell
            else:
                quad.set_color([0.0, 0.0, 0.0, 1.0])
                
            quad.set_edgecolor([0.3, 0.3, 0.3, 1.0])
            ax.add_collection3d(quad)


def plot_cells_distal(ax, tsframe, T03, scalarMap):
    span = 0.5
    for y in range(13):
        for x in range(6):
            vertex = np.empty([4,3])
            vertex[0,:] = DH.get_xyz_distal(x-span, y+span, T03)
            vertex[1,:] = DH.get_xyz_distal(x+span, y+span, T03)
            vertex[2,:] = DH.get_xyz_distal(x+span, y-span, T03)
            vertex[3,:] = DH.get_xyz_distal(x-span, y-span, T03)
            quad = Axes3d.art3d.Poly3DCollection([vertex])

            color = scalarMap.to_rgba(tsframe[y,x])              
            
            if(tsframe[y,x] > 0.0):
                quad.set_color([color[0], color[1], color[2], 1.0]) # Active cell
            else:
                quad.set_color([0.0, 0.0, 0.0, 1.0])
                
            quad.set_edgecolor([0.3, 0.3, 0.3, 1.0])
            ax.add_collection3d(quad)


def preshape_pinch(close_ratio):
    Phi0 =  90 # Rotational axis (Finger 0 + 2)
    # Finger 0
    Phi1 = -72 + close_ratio * 82
    Phi2 =  72 - close_ratio * 82
    # Finger 1
    Phi3 = -90
    Phi4 =  0
    # Finger 2
    Phi5 = -72 + close_ratio * 82
    Phi6 =  72 - close_ratio * 82
    
    return Phi0, Phi1, Phi2, Phi3, Phi4, Phi5, Phi6


profileName = os.path.abspath("bottle_plastic_contact_lens.dsa")
frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileName);

numTSFrames = frameManager.get_tsframe_count();
frameID = 100

tsframe_proximal_f0 = frameManager.get_tsframe(frameID, 0);
tsframe_distal_f0 = frameManager.get_tsframe(frameID, 1);
tsframe_proximal_f1 = frameManager.get_tsframe(frameID, 2);
tsframe_distal_f1 = frameManager.get_tsframe(frameID, 3);
tsframe_proximal_f2 = frameManager.get_tsframe(frameID, 4);
tsframe_distal_f2 = frameManager.get_tsframe(frameID, 5);

# Normalize frames
tsframe_proximal_f0 /= max(1.0, frameManager.get_max_matrix(frameID, 0))
tsframe_distal_f0 /= max(1.0, frameManager.get_max_matrix(frameID, 1))
tsframe_proximal_f1 /= max(1.0, frameManager.get_max_matrix(frameID, 2))
tsframe_distal_f1 /= max(1.0, frameManager.get_max_matrix(frameID, 3))
tsframe_proximal_f2 /= max(1.0, frameManager.get_max_matrix(frameID, 4))
tsframe_distal_f2 /= max(1.0, frameManager.get_max_matrix(frameID, 5))


#######################
# Extract Joint Angles
#######################

#Phi0, Phi1, Phi2, Phi3, Phi4, Phi5, Phi6 = preshape_pinch(0.5) # Simulated grasp

#numAngles = frameManager.get_jointangle_frame_count()
#angleID = 100
#jointangle_frame = frameManager.get_jointangle_frame(angleID)

#corresponding_jointangle_frame = frameManager.get_corresponding_jointangles(frameID)
#theta = frameManager.get_corresponding_jointangles(frameID)

Phi0 = 90 # Rotational axis (Finger 0 + 2) [0, 90]
Phi1 = -15 # Finger 0 proximal [-90, 90]
Phi2 = 15 # Finger 0 distal [-90, 90]
Phi3 = -90 # Finger 1 proximal [-90, 90]
Phi4 = 0 # Finger 1 distal [-90, 90]
Phi5 = -15 # Finger 2 proximal [-90, 90]
Phi6 = 15 # Finger 2 distal [-90, 90]
theta = np.array([Phi0, Phi1, Phi2, Phi3, Phi4, Phi5, Phi6], dtype=np.float64)


# Compute transformation matrices 
T01_f0, T02_f0, T03_f0 = DH.create_transformation_matrices_f0(Phi0, Phi1, Phi2) # Finger 0
T01_f1, T02_f1, T03_f1 = DH.create_transformation_matrices_f1(Phi3, Phi4)       # Finger 1
T01_f2, T02_f2, T03_f2 = DH.create_transformation_matrices_f2(Phi0, Phi5, Phi6) # Finger 2

# Forward kinematics of active sensor cells

# Finger 0
points_f0_prox = project_active_cells_proximal(tsframe_proximal_f0, T02_f0)
points_f0_dist = project_active_cells_distal(tsframe_distal_f0, T03_f0)
# Finger 1
points_f1_prox = project_active_cells_proximal(tsframe_proximal_f1, T02_f1)
points_f1_dist = project_active_cells_distal(tsframe_distal_f1, T03_f1)
# Finger 3
points_f2_prox = project_active_cells_proximal(tsframe_proximal_f2, T02_f2)
points_f2_dist = project_active_cells_distal(tsframe_distal_f2, T03_f2)


# Merge all active sensor cells
points = np.vstack((points_f0_prox, points_f0_dist,
                    points_f1_prox, points_f1_dist,
                    points_f2_prox, points_f2_dist))


# Compute convex hull and diameter
diameter, hull, endpoints_idx = compute_diameter(points)


# Compute Minimal bounding sphere
features = framemanager_python.FeatureExtractionWrapper(frameManager)
#minimal_bounding_sphere = features.compute_minimal_bounding_sphere(frameID, Phi0, Phi1, Phi2, Phi3, Phi4, Phi5, Phi6)
minimal_bounding_sphere = features.compute_minimal_bounding_sphere(frameID, theta)

tx = minimal_bounding_sphere[0]
ty = minimal_bounding_sphere[1]
tz = minimal_bounding_sphere[2]
r = minimal_bounding_sphere[3]



############
# Plotting
###########
text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0

size_factor = 0.75
figure_width = size_factor*text_width
figure_height = (figure_width / golden_ratio)
#figure_height = (text_width / golden_ratio) # height is golden ratio to page width

#figure_height = 1.3 * figure_width
#figure_size = [figure_width, figure_height]
figure_size = [figure_width, figure_width]

config.load_config_small()


# Arbitrary colormap
import matplotlib.colors as colors
import matplotlib.cm as cmx

cdict = {'red': ((0.0, 0.9, 0.9),
                     (1.0, 0.9, 0.9)),
           'green': ((0.0, 0.9, 0.9),
                     (1.0, 0.0, 0.0)),
           'blue':  ((0.0, 0.0, 0.0),
                     (1.0, 0.0, 0.0))}
                     
#plt.register_cmap(name='YELLOW_RED', data=cdict)
#colorMap = plt.get_cmap("YELLOW_RED")

colorMap = plt.get_cmap("YlOrRd_r")
colorNorm = colors.Normalize(vmin=0, vmax=1.0)
scalarMap = cmx.ScalarMappable(norm=colorNorm, cmap=colorMap)



fig = plt.figure(figsize=figure_size, dpi=100)
ax = fig.gca(projection='3d')


# Finger 0
plot_cells_proximal(ax, tsframe_proximal_f0, T02_f0, scalarMap)
plot_cells_distal(ax, tsframe_distal_f0, T03_f0, scalarMap)
# Finger 1
plot_cells_proximal(ax, tsframe_proximal_f1, T02_f1, scalarMap)
plot_cells_distal(ax, tsframe_distal_f1, T03_f1, scalarMap)
# Finger 2
plot_cells_proximal(ax, tsframe_proximal_f2, T02_f2, scalarMap)
plot_cells_distal(ax, tsframe_distal_f2, T03_f2, scalarMap)


# Plot convex hull
#plot_convex_hull(ax, points, hull, endpoints_idx)

# Plot Miniball
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = r * np.outer(np.cos(u), np.sin(v)) + tx
y = r * np.outer(np.sin(u), np.sin(v)) + ty
z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + tz


sphere = ax.plot_surface(x, y, z, rstride=5, cstride=5,
                         color=[0.5, 0.5, 0.5, 0.05], edgecolor=[0.3, 0.3, 0.3, 0.5],
                         linestyle=':', linewidth=0.75, antialiased=True, shade=True)
                
#sphere.set_color(UIBK_blue+[0.0])
#sphere.set_edgecolor(UIBK_blue+[0.3])
#sphere.set_edgecolor([0.3, 0.3, 0.3, 0.3])



#ax.view_init(20, 15) # elevation in z plane, azimuth angle in the x,y plane
#ax.view_init(45, 30) # elevation in z plane, azimuth angle in the x,y plane
ax.azim = -140
ax.elev = 20
ax.dist = 5


#ax.auto_scale_xyz([tx-r, tx+r], [ty-r, ty+r], [tz-r, tz+r])
ax.auto_scale_xyz([-150, 50], [-100, 100], [0, 200])
ax.set_aspect('equal')

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.gca().patch.set_facecolor('white')
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

ax.grid(False)
ax.set_axis_off()

#plt.show() 
fig.tight_layout()

plotname = "miniball"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()
