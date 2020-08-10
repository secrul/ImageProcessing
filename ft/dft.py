# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('../images/lena.png',0)
# """
# DFT_ROWS 设置时，对于每一行单独进行一维傅里叶变换，否则进行二维傅里叶变换或者反傅里叶变换
# DFT_INVERSE 设置时，并且输入室是实数，执行傅里叶变换
# DFT_INVERSE没有被设置，DFT_COMPLEX_OUTPUT 设置时，输出是一个和输入size相同的复数矩阵，没有设置输出一个实数矩阵
# DFT_INVERSE被设置，输入是实数，或者输入是复数DFT_REAL_OUTPUT被设置，输出与输入矩阵等大小的实数阵
# DFT_SCALE 被设置，输出大小会缩放
# """
# dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
#
# dftshift = np.fft.fftshift(dft)
#
# res1 = 20 * np.log(cv2.magnitude(dftshift[:, :, 0], dftshift[:, :, 1]))
#
# # 傅里叶逆变换
# crow, ccol = int(img.shape[0] / 2), int(img.shape[1] / 2) # 求得图像的中心点位置
# mask = np.zeros((img.shape[0], img.shape[1], 2), np.uint8)
# mask[crow-10:crow+10, ccol-10:ccol+10] = 1
#
# # 第六步：将掩模与傅里叶变化后图像相乘，保留中间部分
# mask_img = dftshift * mask
#
# # 第七步：使用np.fft.ifftshift(将低频移动到原来的位置
# img_idf = np.fft.ifftshift(mask_img)
# # dftshift2 = dftshift.copy()
# # dftshift2[95:105,95:105,0] = np.zeros((10,10))
# # ishift = np.fft.ifftshift(dftshift2)
#
# iimg = cv2.idft(img_idf)
#
# res2 = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
#
# # 显示图像
#
# plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
#
# plt.axis('off')
#
# plt.subplot(132), plt.imshow(res1, 'gray'), plt.title('Fourier Image')
#
# plt.axis('off')
#
# plt.subplot(133), plt.imshow(res2, 'gray'), plt.title('Inverse Fourier Image')
#
# plt.axis('off')
#
# plt.show()
"""
author: muzhan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../images/lena.png',0)
# 获取rgb通道中的一个
print(img.shape)
img = np.float32(img)  # 将数值精度调整为32位浮点型
img_dct = cv2.dct(img)  # 使用dct获得img的频域图像

img_recor2 = cv2.idct(img_dct)  # 使用反dct从频域图像恢复出原图像(有损)
plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')

plt.axis('off')

plt.subplot(132), plt.imshow(img_dct, 'gray'), plt.title('Fourier Image')

plt.axis('off')

plt.subplot(133), plt.imshow(img_recor2, 'gray'), plt.title('Inverse Fourier Image')

plt.axis('off')

plt.show()