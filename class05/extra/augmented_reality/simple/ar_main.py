
# Useful links
# http://www.pygame.org/wiki/OBJFileLoader
# https://rdmilligan.wordpress.com/2015/10/15/augmented-reality-using-opencv-opengl-and-blender/
# https://clara.io/library

# TODO -> Implement command line arguments (scale, model and object to be projected)
#      -> Refactor and organize code (proper funcition definition and separation, classes, error handling...)

import argparse

import cv2
import numpy as np
import math
import os
from objloader_simple import *


### Setting printing options
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
np.set_printoptions(precision=3,suppress=True)



#Load the dictionary that was used to generate the markers.
#Initialize the detector parameters using default values

parameters =  cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoDetector = cv2.aruco.ArucoDetector(dictionary, parameters)



# Minimum number of matches that have to be found
# to consider the recognition valid

DEFAULT_COLOR = (123, 100, 255)
dir_name=''

def main():
    """
    This functions loads the target surface image,
    """
    
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    
    # Load 3D model from OBJ file
    obj = OBJ(os.path.join(dir_name, 'models/fox.obj'), swapyz=True)  
    # init video capture
    cap = cv2.VideoCapture(2)

    while True:
        # read the current frame
        ret, frame = cap.read()
        
        #print (frame.shape)
        if not ret:
            print("Unable to capture video")
            return 
        
        # Detect the markers in the image
        markerCorners, ids, rejectedImgPoints = arucoDetector.detectMarkers(frame)
        
        # Check if any marker was detected and insert the image 
        if markerCorners != ():
        
    	    # Estimate the pose of the first ArUco Marker
    	    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners[0], 1, camera_parameters, None)
    	    frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners,ids)
    	    # Build the extrinsic parameter matrix
    	    rmtx = cv2.Rodrigues(rvec)[0]
    	    tmtx = tvec[0]
    	    RT = np.hstack((rmtx,tmtx.T))
    	    
    	    
    	    # obtain 3D projection matrix 
    	    projection = projection_matrix(camera_parameters, RT)
    	    # project cube or model
    	    frame = render(frame, obj, projection, False)
    	    
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
  

    cap.release()
    cv2.destroyAllWindows()
    return 0

def render(img, obj, projection, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    
    # Adjust the value to scale the object
    scale_matrix = np.eye(3)*0.03
    # Adjust the angle to rotate the object
    angle = 90*3.14/180
    Rz = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    
    

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        # Transform the object before projecting
        points = np.dot(points, Rz)
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0], p[1], p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, RT):
    """
    From the camera calibration matrix and the estimated pose of the ArUco marker
    """     
    projection = np.dot(camera_parameters, RT)
    
    return projection
    
def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))



if __name__ == '__main__':
    main()
