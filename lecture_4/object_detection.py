import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('calc.jpg',0)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)

cap = cv2.VideoCapture(0)


while True:
    _, img2 = cap.read()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)
    matches = bf.match(des1,des2)
    
    matches = sorted(matches, key=lambda x: x.distance)
    out = cv2.drawMatches(img1,kp1,img2,kp2,matches[:min(len(matches),15)],None,flags=2)

    cv2.imshow('out', out)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()