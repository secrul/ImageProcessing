import cv2
import numpy as np

# 基本坐标变换

def move(image,Tx,Ty):
    """
    平移变换
    :param image: 输入原图
    :param Tx: x方向偏移
    :param Ty: y方向偏移
    :return: 两个新的画布，体现偏移
    """
    canvas = np.zeros((400,400,3))
    move = [[1,0,Tx],[0,1,Ty],[0,0,1]]
    move = np.array(move)

    canvas2 = canvas.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):
                canvas[i][j][k] = image[i][j][k]

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            source = np.array([i,j,1])
            dst = move.dot(source.T)
            dsti = int(dst[0])
            dstj = int(dst[1])
            if 0 < dsti < 400 and 0 < dstj < 400:
                canvas2[dsti][dstj][0] = image[i][j][0]
                canvas2[dsti][dstj][1] = image[i][j][1]
                canvas2[dsti][dstj][2] = image[i][j][2]
    return canvas,canvas2


def scaling(image,Sx,Sy):
    """
    放缩变换,原理相同，没有放缩操作，当Sx，Sy互为倒数时为拉伸变换
    :param image:
    :param Sx:
    :param Sy:
    :return:
    """
    canvas = np.zeros((400,400,3))
    move = [[Sx, 0, 0], [0, Sy, 0], [0, 0, 1]]
    move = np.array(move)

    canvas2 = canvas.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):
                canvas[i][j][k] = image[i][j][k]

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            source = np.array([i, j, 1])
            dst = move.dot(source.T)
            dsti = int(dst[0])
            dstj = int(dst[1])
            if 0 < dsti < 400 and 0 < dstj < 400:#超出画布范围
                canvas2[dsti][dstj][0] = image[i][j][0]
                canvas2[dsti][dstj][1] = image[i][j][1]
                canvas2[dsti][dstj][2] = image[i][j][2]
    return canvas, canvas2


def rotate(image,a):
    """
    顺时针旋转a
    :param image:
    :param a:
    :return:
    """
    canvas = np.zeros((400,400,3))
    move = [[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]]
    move = np.array(move)

    canvas2 = canvas.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):
                canvas[i][j][k] = image[i][j][k]

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            source = np.array([i, j, 1])
            dst = move.dot(source.T)
            dsti = int(dst[0])
            dstj = int(dst[1])
            if 0 < dsti < 400 and 0 < dstj < 400:#超出画布范围
                canvas2[dsti][dstj][0] = image[i][j][0]
                canvas2[dsti][dstj][1] = image[i][j][1]
                canvas2[dsti][dstj][2] = image[i][j][2]
    return canvas, canvas2


def union(image, Tx, Ty, Sx, Sy, a):
    """
    按照平移、放缩、旋转的顺序
    :param image:
    :param Tx:
    :param Ty:
    :param Sx:
    :param Sy:
    :param a:
    :return:
    """
    canvas = np.zeros((800, 800, 3))
    move1 = [[1, 0, Tx], [0, 1, Ty], [0, 0, 1]]
    move1 = np.array(move1)
    move2 = [[Sx, 0, 0], [0, Sy, 0], [0, 0, 1]]
    move2 = np.array(move2)
    move3 = [[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]]
    move3 = np.array(move3)

    canvas2 = canvas.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):
                canvas[i][j][k] = image[i][j][k]

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            source = np.array([i, j, 1])
            dst = move1.dot(source.T)
            dst = move2.dot(dst.T)
            dst = move3.dot(dst.T)
            dsti = int(dst[0])
            dstj = int(dst[1])
            if 0 < dsti < 800 and 0 < dstj < 800:  # 超出画布范围
                canvas2[dsti][dstj][0] = image[i][j][0]
                canvas2[dsti][dstj][1] = image[i][j][1]
                canvas2[dsti][dstj][2] = image[i][j][2]
    return canvas, canvas2


def shearing(image, Jx = 0, Jy = 0):
    """
    Jx不为0，Jy为0时水平剪切；Jx为0，Jy不为0时垂直剪切
    :param Jx:
    :param Jy:
    :return:
    """
    canvas = np.zeros((400, 400, 3))
    move = [[1, Jx, 0], [Jy, 1, 0], [0, 0, 1]]
    move = np.array(move)

    canvas2 = canvas.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):
                canvas[i][j][k] = image[i][j][k]

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            source = np.array([i, j, 1])
            dst = move.dot(source.T)
            dsti = int(dst[0])
            dstj = int(dst[1])
            if 0 < dsti < 400 and 0 < dstj < 400:  # 超出画布范围
                canvas2[dsti][dstj][0] = image[i][j][0]
                canvas2[dsti][dstj][1] = image[i][j][1]
                canvas2[dsti][dstj][2] = image[i][j][2]
    return canvas, canvas2



img = '../images/lena.png'
image = cv2.imread(img) / 255

#平移操作
# Tx = 100
# Ty = 100
# c1,c2 = move(image,Tx,Ty)
# cv2.imshow('before',c1)
# cv2.imshow('after',c2)
# cv2.waitKey(0)

# # 放缩操作
# Sx = 2
# Sy = 2
# c1,c2 = scaling(image,Sx,Sy)
# cv2.imshow('before',c1)
# cv2.imshow('after',c2)
# cv2.waitKey(0)

#旋转操作
# a = np.pi / 6 # 顺时针旋转
# c1,c2 = rotate(image,a)
# cv2.imshow('before',c1)
# cv2.imshow('after',c2)
# cv2.waitKey(0)

# #先平移，再放缩，最后旋转的级联操作
# Tx = 50
# Ty = 50
# Sx = 2
# Sy = 2
# a = np.pi / 6 # 顺时针旋转
# c1,c2 = union(image, Tx, Ty, Sx, Sy, a)
# cv2.imshow('before',c1)
# cv2.imshow('after',c2)
# cv2.waitKey(0)

# #剪切
# Jx = 1
# Jy = 0
# c1,c2 = shearing(image,Jx,Jy)
# cv2.imshow('before',c1)
# cv2.imshow('after',c2)
# cv2.waitKey(0)
