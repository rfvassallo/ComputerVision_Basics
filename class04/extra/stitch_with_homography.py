# Code created based on
# https://pyimagesearch.com/2016/01/11/opencv-panorama-stitching/


# import the necessary packages

import argparse
import imutils
import cv2
import numpy as np



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="path to the first image")
ap.add_argument("-s", "--second", required=True,
	help="path to the second image")
ap.add_argument("-d", "--direction", type=int, default=1,
	help="stitching direction")
args = vars(ap.parse_args())


# Flag to define direction for stitching images
LEFT2RIGHT = args["direction"]

# load the two images 
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
#imageA = cv2.imread("../images/test04/img2.jpeg")
#imageB = cv2.imread("../images/test04/img3.jpeg")

print (imageA.shape)
print (imageB.shape)

if LEFT2RIGHT == 0: 

    # Get image dimensions
    h, w, c = imageB.shape

    # Create a new blank (black) image with double width
    new_width = w * 2
    new_img = np.zeros((h, new_width, c), dtype=imageB.dtype)

    # Place the original imageB on the right half
    new_img[:, w:] = imageB
    imageB= new_img
    print (imageB.shape)


# convert the image to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# Create SIFT object
descriptor = cv2.SIFT_create()

# Detect keypoints and their feature descriptors
# ImageA
(kpsA, featuresA) = descriptor.detectAndCompute(imageA, None)

# Detect keypoints and their feature descriptors
# ImageB
(kpsB, featuresB) = descriptor.detectAndCompute(imageB, None)

# Create Matcher 
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# The variable search_params specifies the number of times the trees in the index should
# be recursively traversed. Higher values gives better precision, but also takes more time.
search_params = dict(checks = 50)
# FLANN Matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(featuresA,featuresB,k=2)

# Need to draw only good matches, so create a mask
# store all the good matches as per Lowe's ratio test.
good =[]
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
       good.append(m)

# Convert selected keypoints to numpy arrays to be used to compute the homography 
# between images
ptsA = np.float32([ kpsA[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
ptsB = np.float32([ kpsB[m.trainIdx].pt for m in good ]).reshape(-1,1,2)



if LEFT2RIGHT:
    # Compute the homography
    (H, mask) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 4.0)
    matchesMask = mask.ravel().tolist()
    result = cv2.warpPerspective(imageB, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    result[0:imageA.shape[0], 0:imageA.shape[1]] = imageA
    
else:
    # Compute the homography
    (H, mask) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)
    matchesMask = mask.ravel().tolist()
    result = cv2.warpPerspective(imageA, H, (imageB.shape[1], imageA.shape[0]))
    result[:, w:] = imageB[:, w:]


# Generate the image showing the matches
# Draw only the selected matches
img_matches = cv2.drawMatches(
    imageA, kpsA,    # first image and its keypoints
    imageB, kpsB,    # second image and its keypoints
    good, # only the good matches
    None,         # output image (None = new image)
    matchColor=(0, 255, 0),      # matches in green
    singlePointColor= None,# unmatched keypoints in blue
    matchesMask = matchesMask, # matches that satisfy the homography
    flags=2
)



# show the images
cv2.namedWindow('Image A',cv2.WINDOW_NORMAL)
cv2.imshow("Image A", imageA)
cv2.namedWindow('Image B',cv2.WINDOW_NORMAL)
cv2.imshow("Image B", imageB)
cv2.namedWindow('Keypoint Matches',cv2.WINDOW_NORMAL)
cv2.imshow("Keypoint Matches", img_matches)
cv2.namedWindow('Result',cv2.WINDOW_NORMAL)
cv2.imshow("Result", result)
cv2.waitKey(0)


