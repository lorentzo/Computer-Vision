# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 10:26:42 2018

In this project we will see how to work with some popular feature detectors
and descriptors:
    SIFT
    HOG
    HAAR
using openCV

@author: Lovro
"""

import cv2
import numpy as np

# function for displaying image
def imshow(img):
    cv2.namedWindow("me", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("me", 480, 360)
    cv2.imshow("me", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# read image
img = cv2.imread('figures\me.jpg')

# show image
imshow(img)

# transform to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# show gray image
imshow(gray)

############################# SIFT #########################################
sift = cv2.xfeatures2d.SIFT_create()

# detecting features(key points)
key_points = sift.detect(gray, None)
imgKP = cv2.drawKeypoints(gray, key_points, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#display
imshow(imgKP)

# creating descriptor for key points 
descriptors = sift.compute(gray, key_points, img)

# calculating keypoints and descriptors in one step
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)

########################## HOG ###############################################

# function for calculating HOG
def hog(img):
    
    """
    First step: preprocessing. No preprocess steps like:
        -normalization of gamma and color
        -gaussian blur
    are needed for HOG.
    """
    #...
    
    """
    Second step: calculation of gradient. Most common:
        -Sobel (gaussian kernel with differentiation, therefore resistant to noise)
        -1D [-1,0,1] and [-1,0,1]^T
    We are going to use Sobel.
    """
    #Sobel(image, output datatype, dx, dy)
    #result: binary horizontal and vertical edge map, size as original image
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)  
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    
    """
    Third step is to calculate cell histograms.
    Image is split into cells that can be rectangular or circular. 
    Every pixel in cell casts vote for orientation based histogram. Weight of
        vote is proportional to magnitude.
    Histogram has values ranging from 0 to 360 (signed) or 0 to 180 (unsigned)
    """
    bin_n = 16
    magnitude, angle = cv2.cartToPolar(grad_x, grad_y)
    bins = np.int32(bin_n*angle/(2*np.pi))
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = magnitude[:10,:10], magnitude[10:,:10], magnitude[:10,10:], magnitude[10:,10:]
    
    """
    Now we group pixel cells into overlapping blocks.
    This is done because of changes in illumination and contrast so gradient
        strengths could be normalized locally.
    Blocks can be rectangular (R-HOG) or circular (C-HOG).
    R-HOG is described by number of cells in block, number of pixels in cells
        and number of channels per histogram.
    """
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    
    return hist

# find HOG descriptors for image
hog_des = hog(gray)
    
    
    
    
    
    
    

