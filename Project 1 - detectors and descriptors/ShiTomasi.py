# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 09:37:17 2018

@author: Lovro
"""

import cv2
import numpy as np

import common

# read image
img = cv2.imread('figures\\box.jpg')
common.imshow(img, 800, 600)

# convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find corners
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)

#mark corners on image 
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)
    
common.imshow(img)