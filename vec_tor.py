import numpy as np
import cv2
def dot_product(vector1, vector2):
    return sum([v1 * v2 for v1, v2 in zip(vector1, vector2)])


vector1 = [1, 2, 3]
vector2 = [2, 3, 4]
v = np.array([1,2])
w = np.array([2,3])

output = dot_product(vector1, vector2)
#print(output)

#print('method 1: \n', v.dot(w))
#print('method 2: \n', np.dot(v,w))
color_img = cv2.imread('Picture/nature.png',1)
color_img = color_img.dot([0.072, 0.715, 0.213]) # tich vo huong
cv2.imwrite('outputt.jpg', color_img) # chuyen tu rgb sang greyscale

def get_value(u,a):
    v = np.array([0,0,0])
    v[a] = 1
    return u.dot(v)

u = np.array([3,5,7])
a = 0
print(get_value(u,a))