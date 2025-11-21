## Code to read and plot a 3D STL model

# Before using the example code, you will need to install ```numpy-stl``` 

from stl import mesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from math import pi,cos,sin

# Transformation functions

def move (dx,dy,dz):
    T = np.eye(4)
    T[0,-1] = dx
    T[1,-1] = dy
    T[2,-1] = dz
    return T

from math import pi,cos,sin

def z_rotation(angle):
    rotation_matrix=np.array([[cos(angle),-sin(angle),0,0],[sin(angle),cos(angle),0,0],[0,0,1,0],[0,0,0,1]])
    return rotation_matrix

def x_rotation(angle):
    rotation_matrix=np.array([[1,0,0,0],[0, cos(angle),-sin(angle),0],[0, sin(angle), cos(angle),0],[0,0,0,1]])
    return rotation_matrix

def y_rotation(angle):
    rotation_matrix=np.array([[cos(angle),0, sin(angle),0],[0,1,0,0],[-sin(angle), 0, cos(angle),0],[0,0,0,1]])
    return rotation_matrix

# Load the STL files and add the vectors to the plot
your_mesh = mesh.Mesh.from_file('stl_models/donkey_kong.STL')

# Get the x, y, z coordinates contained in the mesh structure that are the
# vertices of the triangular faces of the object
x = your_mesh.x.flatten()
y = your_mesh.y.flatten()
z = your_mesh.z.flatten()

# Get the vectors that define the triangular faces that form the 3D object
obj_vectors = your_mesh.vectors

# Create the 3D object from the x,y,z coordinates and add the additional array of ones to
# represent the object using homogeneous coordinates
obj = np.array([x.T,y.T,z.T,np.ones(x.size)])

#print(obj.shape)

###################################################
# Plotting the 3D vertices of the triangular faces
###################################################

# Create a new plot
fig = plt.figure(figsize=[10,10])
axes0 = plt.axes(projection='3d')

# Plot the points drawing the lines
axes0.plot(obj[0,:],obj[1,:],obj[2,:],'r')
axes0.set_aspect('equal')


###################################################
# Plotting the 3D triangular faces of the object
###################################################

# Create a new plot
fig = plt.figure(figsize=[10,10])
axes1 = plt.axes(projection='3d')

# Plot and render the faces of the object
axes1.add_collection3d(art3d.Poly3DCollection(obj_vectors))
# Plot the contours of the faces of the object
axes1.add_collection3d(art3d.Line3DCollection(obj_vectors, colors='k', linewidths=0.2, linestyles='-'))
# Plot the vertices of the object
#axes1.plot(obj[0,:],obj[1,:],obj[2,:],'k.')

# Set axes and their aspect
axes1.auto_scale_xyz(obj[0,:],obj[1,:],obj[2,:])
axes1.view_init(elev=45,azim=-35)
axes1.dist=10
axes1.set_aspect('equal')

T = move (40,-5,80)
R = x_rotation(pi/4)
R2 = y_rotation(-pi/4)

M = np.dot(R,T)
M2 = R2@R@T

obj2 = np.dot(R,np.dot(T,obj))
# You can also use M to transform the object instead of using R and T separated
# obj2a will be equal to obj2
obj2a = np.dot(M, obj)

obj3 = np.dot(M2,obj)

# Create a new plot
fig = plt.figure(figsize=[10,10])
axes2 = plt.axes(projection='3d')
# Plot the points drawing the lines
axes2.plot(obj[0,:],obj[1,:],obj[2,:],'r')
axes2.plot(obj2[0,:],obj2[1,:],obj2[2,:],'b')
axes2.plot(obj2a[0,:],obj2a[1,:],obj2a[2,:],'*k')
axes2.plot(obj3[0,:],obj3[1,:],obj3[2,:],'g')
axes2.set_aspect('equal')

# Show the plots
plt.show()
