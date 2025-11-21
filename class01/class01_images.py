import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import imutils
import matplotlib
matplotlib.use("TkAgg")


# Read image using OpenCV
img1 = cv.imread('comicsStarWars01.jpg')

# Showing image using OpenCV
cv.imshow("Image 01",img1)
key = cv.waitKey(0)
cv.destroyAllWindows()


# Reading color image as gray
img2 = cv.imread('comicsStarWars01.jpg',0) 



# Converting image to Gray and RGB (OpenCV works with images in BGR format)
gray= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

# Using figure - Matplotlib
fig = plt.figure(figsize=(7,7))
plt.title('Image RGB')
plt.imshow(rgb)

# Using Subfigures - Matplotlib
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(2,2,1)
plt.title('Original image')
plt.imshow(img1)
ax1 = fig.add_subplot(2,2,2)
plt.title('Converting image to RGB to be shown in Matplotlib')
plt.imshow(rgb)
ax1 = fig.add_subplot(2,2,3)
plt.title('Converting image to gray')
plt.imshow(gray,'gray')
ax1 = fig.add_subplot(2,2,4)
plt.title('Reading image as gray')
plt.imshow(img2,'gray')
plt.show()


# Rotating images by multiples of 90 degrees
# cv.ROTATE_90_COUNTERCLOCKWISE
# cv.ROTATE_180

img3 = cv.rotate(rgb,cv.ROTATE_90_CLOCKWISE)

# Using figure - Matplotlib
fig = plt.figure(figsize=(7,7))
plt.title('Rotated image')
plt.imshow(img3)
plt.show()


# Rotating images by 30 degrees in a counter-clockwise direction.

rows,cols,channels= rgb.shape 
M = cv.getRotationMatrix2D((cols/2,rows/2),-30,1) 

img4 = cv.warpAffine(rgb,M,(cols,rows))

# Using figure - Matplotlib
fig = plt.figure(figsize=(7,7))
plt.title('Rotated image')
plt.imshow(img4)
plt.show()



# Extract part of an image using openCV

y = 400
x = 250
h = 200
w = 350

crop_img = rgb[y:y+h, x:x+w]
img5 = rgb

# Paste rotate by 180 degrees
img5[y:y+h, x:x+w] = cv.rotate(crop_img,cv.ROTATE_180)



fig = plt.figure(figsize=(10,5))
plt.xticks([])
plt.yticks([])
ax1 = fig.add_subplot(1,2,1)
plt.title('Cropped image')
plt.imshow(crop_img)
ax1 = fig.add_subplot(1,2,2)
plt.title('New image')
plt.imshow(img5)
plt.show()



