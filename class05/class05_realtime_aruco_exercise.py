

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



#Load the dictionary that was used to generate the markers.
#Initialize the detector parameters using default values

parameters =  cv.aruco.DetectorParameters()
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
arucoDetector = cv.aruco.ArucoDetector(dictionary, parameters)

# Load the image that will replace the ArUco marker 
img = cv.imread('./images/img02.jpg')

# Get the limits of the image that will be inserted in the original one
[l,c,ch] = np.shape(img)
pts_src = np.array([[0,0],[c,0],[c,l],[0,l]])


# Flag to use the camera or a previous recorded video
CAM_ON = False

if CAM_ON :
    # initialize the video stream 
    print("[INFO] starting video stream...")
    cap = cv.VideoCapture(2)
    
else:
    # Load a video from a file
    print("[INFO] reading video file...")
    cap = cv.VideoCapture('./images/output.avi')


#### Complete the code based on the last example where we detected ArUcos makers and replaced 
#### them with other images


# Main Loop
while True:

    # Capture the frame
    ret, frame = cap.read()
    if not ret:
        print('[INFO] No frames. Exiting the program...')
        break
        
    # Detect the markers in the image
    markerCorners, ids, rejectedImgPoints = arucoDetector.detectMarkers(frame)

    # Check if any marker was detected and insert the image 
    if markerCorners != (): 
    
        ### Just take a look on the class05_using_arucos.ipynb or .py
        
        
        
        
       # You will have to change this line 
       im_out = cv.aruco.drawDetectedMarkers(frame, markerCorners,ids)   
                
    else:
        # If no marker was detected, just show the frame 
        im_out = frame
  
        
    # Show the frame 
    cv.imshow("Detecting Aruco", im_out)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


