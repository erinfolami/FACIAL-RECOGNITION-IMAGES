import cv2
import numpy as np

frontal_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


img = cv2.imread("Erin.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = frontal_face.detectMultiScale(gray,1.1,4)

for x,y,w,h in faces:
 dface = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)
 cv2.imshow("detected face",dface)
 cv2.waitKey(0)

