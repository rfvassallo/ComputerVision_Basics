

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



#Load the dictionary that was used to generate the markers.
#Initialize the detector parameters using default values

parameters =  cv.aruco.DetectorParameters()
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
arucoDetector = cv.aruco.ArucoDetector(dictionary, parameters)

# Load the image that will replace the ArUco marker 
img = cv.imread('../images/img02.jpg')

# Get the limits of the image that will be inserted in the original one
[l,c,ch] = np.shape(img)
pts_src = np.array([[0,0],[c,0],[c,l],[0,l]])


# Flag to use the camera or a previous recorded video
CAM_ON = True

if CAM_ON :
    # initialize the video stream 
    print("[INFO] starting video stream...")
    cap = cv.VideoCapture(2)
    
else:
    # Load a video from a file
    print("[INFO] reading video file...")
    cap = cv.VideoCapture('../images/output.avi')


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
    
        # Destiny points are the corners of the first detected marker
        pts_dst = markerCorners[0].reshape(-1, 2)
        # Calculate Homography
        h, status = cv.findHomography(pts_src, pts_dst)
        # Warp source image to destination based on homography
        warped_image = cv.warpPerspective(img, h, (frame.shape[1],frame.shape[0]))  
        # Prepare a mask representing region to copy from the warped image into the original frame.
        mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8)
        # Create a white polygon in the mask to mark the place to insert the image
        cv.fillConvexPoly(mask, np.int32([pts_dst]), (255, 255, 255), cv.LINE_AA)
        # Erode the mask to not copy the boundary effects from the warping
        element = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
        mask = cv.erode(mask, element, iterations=3)
        # Invert the mask for the region you’re overlaying
        mask_inv = cv.bitwise_not(mask)    
        # Black out the region on the original frame
        frame_bg = cv.bitwise_and(frame, frame, mask=mask_inv)
        # Take only the region from the warped image
        warped_fg = cv.bitwise_and(warped_image, warped_image, mask=mask)
        # Combine the original image with the warped image
        im_out = cv.add(frame_bg, warped_fg)
               
                
    else:
        # If no marker was detected, just show the frame 
        im_out = frame
  
        
    # Show the frame 
    cv.imshow("Detecting Aruco", im_out)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


