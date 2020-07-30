import cv2
import numpy as np

lenaPath = '../images/lena.png'
lena = cv2.imread(lenaPath) / 255
lena = lena.astype(np.float32)
cv2.imshow('lena',lena)
cv2.waitKey(0)
gray = cv2.cvtColor(lena * 255, cv2.COLOR_BGR2GRAY).astype(np.uint8)
cv2.imshow('gray',gray)
cv2.waitKey(0)
dst = cv2.equalizeHist(gray)
cv2.imshow('hist',dst)
cv2.waitKey(0)

one = np.ones_like(lena)
cv2.subtract(one,lena,lena)# y = 1 - x
cv2.imshow('平方',lena)
cv2.waitKey(0)

