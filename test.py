import cv2
import numpy as np

img = cv2.imread('Picture/nature.png', 0)
print(img.shape)
print(img.dtype)

img = img.astype(float)
print(img.dtype)

# increase brightness
img = img + 30
img = np.clip(img,0,255)

img = img.astype(np.uint8)
print(img.dtype)
cv2.imwrite('output.jpg', img)

# axis 0: dòng, hàng
# axis 1: cột
