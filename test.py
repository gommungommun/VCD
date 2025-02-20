import numpy as np
import cv2
import matplotlib.pyplot as plt
import Hyperopia
import Myopia

test_image = cv2.imread('snellen.png')

# PSF 생성
gaussian_filter = gaussian_psf(size=28, sigma=1)  # MNIST 이미지 크기에 맞춰 size=28
airy_filter = airy_psf(size=28)

# 필터 적용
gaussian_result = convolve2d(test_image, gaussian_filter, mode='same')
airy_result = convolve2d(test_image, airy_filter, mode='same')

plt.resize(gaussian_result, '480,640')
plt.resize(gaussian_result, '480,640')

# 결과 시각화
plt.figure(figsize=(15,5))

plt.subplot(131)
plt.imshow(test_image, cmap='gray')
plt.title('Original Image')

plt.subplot(132)
plt.imshow(gaussian_result, cmap='gray')
plt.title('Gaussian PSF Applied')

plt.subplot(133)
plt.imshow(airy_result, cmap='gray')
plt.title('Airy PSF Applied')

plt.show()

# 필터 시각화
plt.figure(figsize=(10,5))

plt.subplot(121)
plt.imshow(gaussian_filter, cmap='gray')
plt.title('Gaussian PSF')

plt.subplot(122)
plt.imshow(airy_filter, cmap='gray')
plt.title('Airy PSF')

plt.show()