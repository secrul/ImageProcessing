import cv2
import numpy as np
import torch.nn.functional as F
import torch


#对比度的计算
#峰值信噪比的计算
#结构相似性的计算


def adjacentAverageFiltering(image):
    # dst = cv2.filter2D(src, ddepth, kernel, [dst, anchor, delta, borderType])
    # cv2.blur
    """
    src:源图
    ddepth  结过图的数据深度，-1代表和原图一样
    """
    image = np.float32(image)
    kernal = np.ones((3,3),np.float32)/9
    dst = cv2.filter2D(image,-1,kernal)
    return dst

def boxFilter(image):
    """
    当normalize = 0卷积和不进行平均，超过255按照255显示，否则计算均值，默认为1
    :param image:
    :return:
    """
    # dst= cv2.boxFilter( src, ddepth, ksize, anchor, normalize, borderType)
    image = np.float32(image)
    dst = cv2.boxFilter(image, -1,(5,5),normalize=0)
    return dst


def gaussianAverageFilter(image):
    # dst = cv2.GaussianBlur(src,ksize,sigmaX ,sigmaY,borderType)
    image = np.float32(image)
    dst = cv2.GaussianBlur(image,ksize=(3,3),sigmaX=2,sigmaY=1)
    return dst


def LaplaceFilter(image):
    image = np.float32(image)
    dst = cv2.Laplacian(image,-1, ksize=3)
    return dst


def notSharpFilter(image):
    image = np.float32(image)
    averageImage = adjacentAverageFiltering(image)
    NSFilter = cv2.subtract(image,averageImage)

    sharpImage = cv2.add(NSFilter,image)
    return NSFilter, sharpImage

imgPath = "../images/lena.png"
lena = cv2.imread(imgPath) / 255
cv2.imshow('lena',lena)

smoothLena = adjacentAverageFiltering(lena)
cv2.imshow('adjacent',smoothLena)

boxLena = boxFilter(lena)
cv2.imshow('box',boxLena)

smoothLena = gaussianAverageFilter(lena)
cv2.imshow('gaussian',smoothLena)

sharpLena = LaplaceFilter(lena)
cv2.imshow('Laplace',sharpLena)

NSimage, sharpLena2 = notSharpFilter(lena)
cv2.imshow('NSimage',NSimage)
cv2.imshow('sharpLena2',sharpLena2)

cv2.waitKey(0)