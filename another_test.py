import cv2
import numpy as np

# 0: Khi bạn đặt nó là 0, nó có nghĩa là hình ảnh sẽ được đọc dưới dạng ảnh xám (grayscale).
# Nếu bạn để nó là 1 hoặc một số khác, hình ảnh sẽ được đọc dưới dạng màu (RGB).
img = cv2.imread('Picture/image.jpg', 1)

x_0, y_0 = 470, 120
x_1, y_1 = 800, 850
color = np.array([0,0,255])
# axis 0 = height = y1-y0
# axis 1 = width = x1 - x0
# blue,green,red
img[y_0, x_0:x_1, :] = color
img[y_1, x_0:x_1, :] = color
img[y_0:y_1, x_0, :] = color
img[y_0:y_1, x_1, :] = color


# cv2.imwrite('image_3.jpg', img)