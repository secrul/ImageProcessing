import cv2
import numpy as np


img = '../images/sheep.png'
image = cv2.imread(img) / 255
img2 = '../images/lena.png'
image2 = cv2.imread(img2) / 255
ansAdd = image / 2 + image2  #小羊灰度值太大，会挡住lena，将小羊的灰度值减半
# cv2.add
cv2.imshow('sheep',image)
cv2.waitKey(0)

cv2.imshow('lena',image2)
cv2.waitKey(0)

cv2.imshow('add',ansAdd)
cv2.waitKey(0)

ansSub = image2 - image
# cv2.subtract
cv2.imshow('sub',ansSub)
cv2.waitKey(0)

ansmul = image2* image
cv2.imshow('mul',ansmul)
cv2.waitKey(0)

ansDiv = image2/(image + 10e-6)
cv2.imshow('div',ansDiv)
cv2.waitKey(0)

image2 = image2.astype(np.float32)
image = image.astype(np.float32)
graylena =  cv2.cvtColor(image2 * 255,cv2.COLOR_BGR2GRAY).astype(np.uint8)
graysheep =  cv2.cvtColor(image  * 255,cv2.COLOR_BGR2GRAY).astype(np.uint8)

# 二值化
for i in range(graysheep.shape[0]):
    for j in range(graysheep.shape[1]):
        if graysheep[i][j] < 128:
            graysheep[i][j] = 0
        else:
            graysheep[i][j] = 255
for i in range(graylena.shape[0]):
    for j in range(graylena.shape[1]):
        if graylena[i][j] < 128:
            graylena[i][j] = 0
        else:
            graylena[i][j] = 255

comp = np.zeros_like(graylena)
cv2.bitwise_not(graylena,comp)
cv2.imshow('not',comp)
cv2.waitKey(0)

andd = np.zeros_like(graylena)
cv2.bitwise_and(graylena,graysheep,andd)
cv2.imshow('and',andd)
cv2.waitKey(0)

orr = np.zeros_like(graylena)
cv2.bitwise_or(graylena,graysheep,orr)
cv2.imshow('or',orr)
cv2.waitKey(0)

xorr = np.zeros_like(graylena)
cv2.bitwise_xor(graylena,graysheep,xorr)
cv2.imshow('xor',xorr)
cv2.waitKey(0)

def addGaussianNoise(image, noise_sigma):
    """
    添加高斯噪声

    :param image:
    :param noise_sigma: 噪声的灰度范围
    :return:
    """
    temp_image = np.float64(np.copy(image))

    h,w,_ = temp_image.shape
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
        noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
        noisy_image[:, :, 2] = temp_image[:, :, 2] + noise

    return noisy_image


noise_sigma = 10
noise_img = addGaussianNoise(image2 * 255, noise_sigma=noise_sigma)
cv2.imshow("gaussianNoise",noise_img / 255)
cv2.waitKey(0)