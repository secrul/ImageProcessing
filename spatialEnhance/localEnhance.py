import cv2
import numpy as np
import matplotlib.pyplot as plt


def localhist(image, k =10):
    """

    :param image:
    :param k:
    :return:
    """
    fun = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))
    image = np.uint8(image)
    tmp = np.zeros_like(image)
    print(image.shape)
    h,w = image.shape
    i = 0
    while i < h:
        j = 0
        while j < w:
            print(i)
            endi = i + k if i + k < h else h
            endj = j + k if j + k < w else w
            tmp[i:endi,j:endj] = cv2.equalizeHist(image[i:endi,j:endj])
            j = endj
        i += k
    return tmp,fun.apply(image)


path = '../images/lena.png'
lena = cv2.imread(path,0)
cv2.imshow('lena',lena)
ans1,ans2 = localhist(lena)
ans3 = cv2.equalizeHist(lena)
cv2.imshow('ans1',ans1)
cv2.imshow('ans2',ans2)
cv2.imshow('ans3',ans3)
plt.subplot(141)
plt.hist(lena.ravel(), 256)
plt.subplot(142)
plt.hist(ans1.ravel(), 256)
plt.subplot(143)
plt.hist(ans2.ravel(), 256)
plt.subplot(144)
plt.hist(ans3.ravel(), 256)
plt.show()
cv2.waitKey(0)