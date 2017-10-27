# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 16:26:31 2017

@author: AT522
"""
###########################################
# gradient and edge detection avec OpenCV #
###########################################

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read() # on lit la trame. "_" car on ne s'occupe pas ici de la 1ere valeur

# on applique un gradient de transformée de Laplace
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)

# on applique un gradient de "sobel" , horizontal ou vertical
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)

# edge detector built in fonction
    edges = cv2.Canny(frame, 100, 200)
   
    cv2.imshow('original', frame)
    cv2.imshow('Laplace', laplacian)
#    cv2.imshow('sobelx', sobelx)
#    cv2.imshow('sobely', sobely)
    cv2.imshow('edge detec', edges)
    
# gestion clavier pour sortir     
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
    
    
    

