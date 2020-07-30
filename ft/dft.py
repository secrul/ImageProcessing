import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../images/lena.png',0)
f = np.fft.fft2(img)

"""
DFT_ROWS 设置时，对于每一行单独进行一维傅里叶变换，否则进行二维傅里叶变换或者反傅里叶变换
DFT_INVERSE 设置时，并且输入室是实数，执行傅里叶变换
DFT_INVERSE没有被设置，DFT_COMPLEX_OUTPUT 设置时，输出是一个和输入size相同的复数矩阵，没有设置输出一个实数矩阵
DFT_INVERSE被设置，输入是实数，或者输入是复数DFT_REAL_OUTPUT被设置，输出与输入矩阵等大小的实数阵
DFT_SCALE 被设置，输出大小会缩放
"""
fshift = np.fft.fftshift(f)#将低频亮区域移到中间
print(type(fshift))
print(fshift.shape)
magnitude_spectrum = 20*np.log(np.abs(fshift))
print(np.max(magnitude_spectrum))
print(np.min(magnitude_spectrum))
# f = cv2.dft(img,flags=cv2.DFT_COMPLEX_OUTPUT) cv2进行傅里叶变换
#fshift = np.fft.fftshift(f)#将低频亮区域移到中间
#该函数用来计算二维矢量的幅值
#img_dft = 20 * np.log(cv2.magnitude(dft_img_ce[:, :, 0], dft_img_ce[:, :, 1]))

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Input Image'),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum,cmap = 'gray')
plt.title('Magnitude Spectrum'),plt.xticks([]),plt.yticks([])
plt.show()