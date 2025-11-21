# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
from imutils.video import VideoStream
import imutils
import time


img_path = "./images/"

# List of images that will replace the ArUco markers
img_list = [img_path+"img01.png", img_path+"img02.png", img_path+"img03.png", img_path+"img04.png"  ]

    
    
img_puzzle = []

for img_file in img_list:
    img = cv2.imread(img_file)
    if img is None:
        print(f"⚠️ Warning: Could not load image {img_file}")
    elif not isinstance(img, np.ndarray):
        print(f"⚠️ Error: {img_file} is not a valid NumPy array.")
    else:
        img_puzzle.append(img)

print(f"✅ Loaded {len(img_puzzle)} images successfully.")    

# List of ArUco markers ID
id_list = [5, 6, 7, 8 ]


# Get the limits of the image that will be inserted in the original one
[l,c,ch] = np.shape(img_puzzle[0])
pts_src = np.array([[0,0],[c,0],[c,l],[0,l]])

#print(pts_src)


#Load the dictionary that was used to generate the markers.
#Initialize the detector parameters using default values

parameters =  cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoDetector = cv2.aruco.ArucoDetector(dictionary, parameters)




# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")


# Open the device at the ID 0
cap = cv2.VideoCapture(2)
# Check whether user selected camera is opened successfully.
if not (cap.isOpened()):
    print("Could not open video device")

# To set the resolution
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)     #640
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)    #480



while(True):
    
    
    # grab the frame from the threaded video stream 
    ret, frame = cap.read()
        
    # Detect the markers in the image
    markerCorners, ids, rejectedImgPoints = arucoDetector.detectMarkers(frame)
    

    # Detect the markers
    img_corners = cv2.aruco.drawDetectedMarkers(frame, markerCorners, ids)

    # If any marker was found
    if markerCorners != (): 
        
        
        for i in range(len(ids)):
        
            # Destiny points are the corners of the marker scaled by a factor defined by scl
            center = np.mean(markerCorners[i].reshape(-1, 2),axis=0)
            scl = 1.18
                    
            pts_dst = markerCorners[i].reshape(-1, 2)
            
            # Scale matrix
            S = np.array([[scl, 0],[0, scl]]) 
            
            #Scale the area of the marker around its center
            aux = np.tile((center), (4, 1)).T
            pts_dst = (S@(pts_dst.T-aux)+aux).T
            
            
            # Select the image of the puzzle according to the marker id
            if ids[i] in id_list:
                img_index = id_list.index(ids[i])
                img = img_puzzle[img_index]
            
            
            # Calculate Homography
            h, status = cv2.findHomography(pts_src, pts_dst)

            # Warp source image to destination based on homography
            warped_image = cv2.warpPerspective(img, h, (frame.shape[1],frame.shape[0]))
            
            
            # Prepare a mask representing region to copy from the warped image into the original frame.
            mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8)
            
            # Create a white polygon in the mask to mark the place to insert the image
            cv2.fillConvexPoly(mask, np.int32([pts_dst]), (255, 255, 255), cv2.LINE_AA)
            
            # Erode the mask to not copy the boundary effects from the warping
            # Erode the mask to not copy the boundary effects from the warping
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            mask = cv2.erode(mask, element, iterations=3)
            
            # Invert the mask for the region you’re overlaying
            mask_inv = cv2.bitwise_not(mask)
            
            # Black out the region on the original frame
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            
            # Take only the region from the warped image
            warped_fg = cv2.bitwise_and(warped_image, warped_image, mask=mask)
            
            # Combine the original image with the warped image
            im_out = cv2.add(frame_bg, warped_fg)
            
            frame = im_out
        
    cv2.namedWindow('Puzzle',cv2.WINDOW_NORMAL)
    cv2.imshow('Puzzle',frame)
    #cv2.resizeWindow('Puzzle', 1280, 800)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

