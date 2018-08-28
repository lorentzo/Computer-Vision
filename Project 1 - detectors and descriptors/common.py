# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 08:38:26 2018

This file contains common functions for computer vision

@author: Lovro
"""

import cv2

# function for displaying image
def imshow(img, hight=640, width=480, name="me",wait=0):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, hight, width)
    cv2.imshow(name, img)
    cv2.waitKey(wait)
    cv2.destroyAllWindows()