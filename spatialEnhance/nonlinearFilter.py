import cv2
import numpy as np


def medFilter(image):
    image = np.float32(image)
    dst = cv2.medianBlur(image,ksize=3)
    return dst


def islegal(i,j,x,y):
    """
    判断图像的坐标是否合法
    :param i:
    :param j:
    :param x:
    :param y:
    :return:
    """
    if 0 < i < x and 0 < j < y:
        return True
    else:
        return False

def percentFilter(image, k = 3, type = 1):
    """
    序统计滤波，最大、最小、中点
    :param image:
    :param k:滤波器大小
    :param type:1-最大值、2-最小值、3-中点
    :return:
    """
    tmpImage = np.zeros_like(image)
    h, w, _ = image.shape
    for i in range(h):
        for j in range(w):
            for k in range(3):
                valueList = [image[i][j][k]]
                if islegal(i-1,j-1,h,w):
                    valueList.append(image[i - 1][j - 1][k])
                if islegal(i - 1, j, h, w):
                    valueList.append(image[i - 1][j][k])
                if islegal(i - 1, j + 1, h, w):
                    valueList.append(image[i - 1][j + 1][k])
                if islegal(i, j - 1, h, w):
                    valueList.append(image[i][j - 1][k])
                if islegal(i, j + 1, h, w):
                    valueList.append(image[i][j + 1][k])
                if islegal(i + 1, j - 1, h, w):
                    valueList.append(image[i + 1][j - 1][k])
                if islegal(i + 1, j, h, w):
                    valueList.append(image[i + 1][j][k])
                if islegal(i + 1, j + 1, h, w):
                    valueList.append(image[i + 1][j + 1][k])
                if type == 1:
                    tmpImage[i][j][k] = max(valueList)
                elif type == 2:
                    tmpImage[i][j][k] = min(valueList)
                elif type == 3:
                    tmpImage[i][j][k] = min(valueList) + max(valueList) / 2
                else:
                    print("滤波器类型错误")
                    return None
    return tmpImage


def gradientSharp(image):
    filter1 = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    filter2 = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    #对于边缘的一个像素不处理
    tmpimage = np.zeros_like(image)
    maxx = 0
    minn = 1
    h,w,_ = image.shape
    for i in range(1,h-1):
        for j in range(1,w-1):
            for k in range(3):
                t = image[i-1:i+2,j-1:j+2,k]
                gx = np.sum(filter1 * t) / 3
                gy = np.sum(filter2 * t) / 3
                tmpimage[i][j][k] = np.sqrt(gx ** 2 + gy ** 2)
                maxx = max(maxx, tmpimage[i][j][k])
                minn = min(minn, tmpimage[i][j][k])
    tmpimage = (tmpimage) / (maxx - minn)
    # print(np.max(tmpimage))
    # print(np.min(tmpimage))
    return tmpimage

def maxminFilter(image, k = 3):
    """
    序统计滤波，最大、最小、中点
    :param image:
    :param k:滤波器大小
    :param type:1-最大值、2-最小值、3-中点
    :return:
    """
    tmpImage = np.zeros_like(image)
    h, w, _ = image.shape
    for i in range(h):
        for j in range(w):
            for k in range(3):
                valueList = [image[i][j][k]]
                if islegal(i-1,j-1,h,w):
                    valueList.append(image[i - 1][j - 1][k])
                if islegal(i - 1, j, h, w):
                    valueList.append(image[i - 1][j][k])
                if islegal(i - 1, j + 1, h, w):
                    valueList.append(image[i - 1][j + 1][k])
                if islegal(i, j - 1, h, w):
                    valueList.append(image[i][j - 1][k])
                if islegal(i, j + 1, h, w):
                    valueList.append(image[i][j + 1][k])
                if islegal(i + 1, j - 1, h, w):
                    valueList.append(image[i + 1][j - 1][k])
                if islegal(i + 1, j, h, w):
                    valueList.append(image[i + 1][j][k])
                if islegal(i + 1, j + 1, h, w):
                    valueList.append(image[i + 1][j + 1][k])
                tmpImage[i][j][k] = min(valueList) if tmpImage[i][j][k] < (max(valueList) + min(valueList)) / 2 \
                    else max(valueList)
    return tmpImage

imgPath = "../images/sheep.png"
lena = cv2.imread(imgPath) / 255
cv2.imshow('lena',lena)

# medlena = medFilter(lena)
# cv2.imshow('med',medlena)

# image = np.float32(lena)
# kernal = np.ones((3, 3), np.float32) / 9
# dst = cv2.filter2D(image, -1, kernal)
# cv2.imshow('aver',dst)

# maxlena = percentFilter(lena,3,1)
# cv2.imshow('maxFilter',maxlena)
#
# minlena = percentFilter(lena,3,2)
# cv2.imshow('minFilter',minlena)
#
# mlena = percentFilter(lena,3,3)#中点滤波
# cv2.imshow('mFilter',mlena)

# mlena = gradientSharp(lena)
# cv2.imshow('gFilter',mlena)
for i in range(10):
    lena = maxminFilter(lena)
    cv2.imshow('mmFilter' + str(i),lena)
cv2.waitKey(0)