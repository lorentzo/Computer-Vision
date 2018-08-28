# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 08:28:36 2018

@author: Lovro
"""

#pro packages
import cv2

#my packages
import common

#read image
img = cv2.imread('figures\\box.jpg')

#display image
common.imshow(img, 800, 600)

#rgb2gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#display
common.imshow(gray,800,600)

#corner detection
corner = cv2.cornerHarris(gray, 2,3,0.04)
#make corner more visible
corner = cv2.dilate(corner, None)
#add corner to original image
img_corner = img
img_corner[corner>0.01*corner.max()]=[0,255,0]

#display
common.imshow(img_corner, 800, 600)