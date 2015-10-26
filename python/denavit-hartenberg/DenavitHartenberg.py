 # -*- coding: utf-8 -*-
import numpy as np

# Generate generic transformation matrix from frame n-1 to frame n
def create_transformation_matrix(d_n, Theta_n, r_n, alpha_n):
    return np.array([ [np.cos(Theta_n), -np.sin(Theta_n)*np.cos(alpha_n),  np.sin(Theta_n)*np.sin(alpha_n), r_n*np.cos(Theta_n)],
                     [np.sin(Theta_n),  np.cos(Theta_n)*np.cos(alpha_n), -np.cos(Theta_n)*np.sin(alpha_n), r_n*np.sin(Theta_n)],
                     [0.0, np.sin(alpha_n), np.cos(alpha_n), d_n],
                     [0.0, 0.0, 0.0, 1.0] ], dtype=np.float64)
 

def create_transformation_matrices_f0(Phi0, Phi1, Phi2):
    
    # Denavit-Hartenberg parameters
    l = np.array([16.7, 86.5, 68.5])
    d = np.array([l[0], 0.0, 0.0])
    Theta = np.array([-np.radians(Phi0), np.radians(Phi1)-np.radians(90.0),  np.radians(Phi2)])
    r = np.array([0.0, l[1], l[2]])
    alpha = np.array([np.radians(-90.0), 0.0, 0.0])

    # Finger transformation matrices
    T01_f0 = create_transformation_matrix(d[0], Theta[0], r[0], alpha[0])
    T12_f0 = create_transformation_matrix(d[1], Theta[1], r[1], alpha[1])
    T23_f0 = create_transformation_matrix(d[2], Theta[2], r[2], alpha[2])

    # Apply finger offset
    d = 66.0
    h = 0.5 * np.sqrt(3) * d
    tx = 1.0/3.0 * h
    ty = -33
    tz = 0.0
    T00_f0 = np.array([ [-1.0, 0.0, 0.0, tx],
                        [0.0, -1.0, 0.0, ty],
                        [0.0, 0.0, 1.0,  tz],
                        [0.0, 0.0, 0.0, 1.0] ], dtype=np.float64)
          
    T01_f0 = T00_f0.dot(T01_f0)     
    T02_f0 = T01_f0.dot(T12_f0) # Base frame for proximal sensor matrix
    T03_f0 = T02_f0.dot(T23_f0) # Base frame for distal sensor matrix (End effector, i.e. fingertip)
   
    return T01_f0, T02_f0, T03_f0


def create_transformation_matrices_f1(Phi3, Phi4):
    
    # Denavit-Hartenberg parameters
    l = np.array([16.7, 86.5, 68.5])
    d = np.array([l[0], 0.0, 0.0])
    Theta = np.array([np.radians(180), np.radians(Phi3)-np.radians(90.0),  np.radians(Phi4)])
    r = np.array([0.0, l[1], l[2]])
    alpha = np.array([np.radians(-90.0), 0.0, 0.0])

    # Finger transformation matrices
    T01_f1 = create_transformation_matrix(d[0], Theta[0], r[0], alpha[0]) # Stiffened joint (Calculate only once)
    T12_f1 = create_transformation_matrix(d[1], Theta[1], r[1], alpha[1])
    T23_f1 = create_transformation_matrix(d[2], Theta[2], r[2], alpha[2])

    # Apply finger offset
    d = 66.0
    h = 0.5 * np.sqrt(3) * d
    tx = -2.0/3.0 * h
    ty = 0.0
    tz = 0.0
    T00_f1 = np.array([ [-1.0, 0.0, 0.0, tx],
                        [0.0, -1.0, 0.0, ty],
                        [0.0, 0.0, 1.0,  tz],
                        [0.0, 0.0, 0.0, 1.0] ], dtype=np.float64)
          
    T01_f1 = T00_f1.dot(T01_f1) # Stiffened joint (Calculate only once)
    T02_f1 = T01_f1.dot(T12_f1) # Base frame for proximal sensor matrix
    T03_f1 = T02_f1.dot(T23_f1) # Base frame for distal sensor matrix (End effector, i.e. fingertip)
   
    return T01_f1, T02_f1, T03_f1



def create_transformation_matrices_f2(Phi0, Phi5, Phi6):
    
    # Denavit-Hartenberg parameters
    l = np.array([16.7, 86.5, 68.5])
    d = np.array([l[0], 0.0, 0.0])
    Theta = np.array([+np.radians(Phi0), np.radians(Phi5)-np.radians(90.0),  np.radians(Phi6)])
    r = np.array([0.0, l[1], l[2]])
    alpha = np.array([np.radians(-90.0), 0.0, 0.0])

    # Finger transformation matrices
    T01_f2 = create_transformation_matrix(d[0], Theta[0], r[0], alpha[0])
    T12_f2 = create_transformation_matrix(d[1], Theta[1], r[1], alpha[1])
    T23_f2 = create_transformation_matrix(d[2], Theta[2], r[2], alpha[2])

    # Apply finger offset
    d = 66.0
    h = 0.5 * np.sqrt(3) * d
    tx = 1.0/3.0 * h
    ty = 33
    tz = 0.0
    T00_f2 = np.array([ [-1.0, 0.0, 0.0, tx],
                       [0.0, -1.0, 0.0, ty],
                       [0.0, 0.0, 1.0,  tz],
                       [0.0, 0.0, 0.0, 1.0] ], dtype=np.float64)
          
    T01_f2 = T00_f2.dot(T01_f2)     
    T02_f2 = T01_f2.dot(T12_f2) # Base frame for proximal sensor matrix
    T03_f2 = T02_f2.dot(T23_f2) # Base frame for distal sensor matrix (End effector, i.e. fingertip)
   
    return T01_f2, T02_f2, T03_f2
    
    
# Sensor matrix coordinates (x,y) -> O2 (x2, y2, z2)
def create_P_prox():
    l2 = 86.5
    s1 = 17.5
    a = 4.1
    w = 6*3.4
    d = 15.43
 
    # Transformation within sensor matrix
    sx =  3.4
    sy = -3.4
    tx = 3.4/2.0 - w/2.0
    ty = 13*3.4
    
    # Translation relative to O2
    tx2 = -l2 + s1 + a
    ty2 = d

    return np.array([[ sy, 0.0, 0.0, ty+tx2],
                     [0.0, 1.0, 0.0,  ty2  ],
                     [0.0, 0.0,  sx,   tx  ],
                     [0.0, 0.0, 0.0,  1.0  ] ], dtype=np.float64)
                        


def create_P_dist(y):
  
    l3 = 68.5
    s3 = 17.5
    a = 4.95
    taxel_width = 3.4
    matrix_width = 6*3.4
    R = 60.0
    d = 15
    
    if(y > 8.5): # Planar part
        # Transformation within sensor matrix
        sx = taxel_width 
        sy = -taxel_width
        tx = taxel_width/2.0 - matrix_width/2.0
        ty = 12*3.4
    
        # Translation relative to O3
        tx3 = -l3 + s3 + a
        ty3 = d
    
    else: # Curved part
        # Transformation within sensor matrix
        sx = taxel_width
        sy = 1.0
        tx = taxel_width/2.0 - matrix_width/2.0
        ty = (R*np.sin( ((8.5-y)*3.4) / R )) + 3.5*3.4 - y    
                
        # Translation relative to O3
        tx3 = -l3 + s3 + a
        ty3 = d - (R-R*np.cos( ((8.5-y)*3.4) / R))
  
  
    return np.array([[ sy, 0.0, 0.0, ty+tx3],
                     [0.0, 1.0, 0.0,  ty3  ],
                     [0.0, 0.0,  sx,   tx  ],
                     [0.0, 0.0, 0.0,  1.0  ] ], dtype=np.float64)



def get_xyz_proximal(x, y, T02):
   
    P_prox = create_P_prox()
    T_total = T02.dot(P_prox)
    
    # Map cell index to point in sensor matrix space
    p = np.array([y, 0.0, x, 1.0])
        
    # Transform point    
    return T_total.dot(p)[0:3] # remove w


def get_xyz_distal(x, y, T03):
   
    P_dist = create_P_dist(y)
    T_total = T03.dot(P_dist)
   
    # Map cell index to point in sensor matrix space
    p = np.array([y, 0.0, x, 1.0])
    
    # Transform point to finger space
    return T_total.dot(p)[0:3] # remove w
