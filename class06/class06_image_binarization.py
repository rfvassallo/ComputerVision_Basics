import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


 
img = cv.imread('./images/stuff.jpg', cv.IMREAD_GRAYSCALE)


assert img is not None, "file could not be read"

# Image blurring to reduce noise 
img = cv.medianBlur(img,5)


# Global Thresholding 
ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# Adaptive Thresholding - Mean 
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,9,5)

# Adaptive Thresholding - Gaussian
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,9,5)
 
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
 
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
