# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 16:26:31 2017

@author: AT522
"""
##################################
# Haar cascades object detection #
##################################

import cv2
import numpy as np

# telechargement des fichiers cascades à partir du lien suivant : 
# https://github.com/Itseez/opencv/tree/master/data/haarcascades
# on prend "frontal face" et "eye"
visage_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
oeil_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0) # on capture les trames de la webcam

while True:
    _, frame = cap.read() # on lit la trame. "_" car on ne s'occupe pas ici de la 1ere valeur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # on convertit l'image en niveaux de gris
    
# on détecte les visages    
    visages = visage_cascade.detectMultiScale(gray, 1.3, 5)   
    
# on trace un rectangle la ou un visage est détecté    
    for (x,y,w,h) in visages:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
# on cherche des yeux quand un visage est détecté (pas besoin de repartir sur
# l'image complete. On reste dans la région de l'image (roi). 
        yeux = oeil_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in yeux:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
# on affiche la video    
    cv2.imshow('frame', frame)

# gestion clavier pour sortir     
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
    
    
    

