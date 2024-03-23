import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Picture/nature.png', 1)
img = img.astype(float)
height,width,depth = img.shape

transform = np.array([[1,0], [0,-1]])
# xoay ngang là [[-1,0],[0,1]]
output = np.zeros((height,width,depth))
for i in range(height):
    for j in range(width):
        pixel = img[i,j,:]

        new_j, new_i = transform.dot(np.array([j,i])) + [0, height - 1] # xoay ngang là + [width-1, 0]

        output[new_i, new_j, :] = pixel

output = output.astype(np.uint8)

cv2.imwrite('output01.jpg', img)
