import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('dataset/training_set/cats/cat.8.jpg')

#show image
fig = plt.figure()
ax = fig.add_subplot(1,3,1)
img_plot = plt.imshow(img)
ax.set_title("img")
print(img_plot)
#show histogram
ax = fig.add_subplot(1,3,2)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
ax.set_title("histogram")

#reduce image colors using K-means
k_color = 3

#image edges using canny method
ax = fig.add_subplot(1,3,3)
ax.set_title("edge")
#blur = cv2.GaussianBlur(img, (5,5), 1)
#cv2.imshow('blur',blur)
edge = cv2.Canny(img,100,200)
plt.imshow(edge)
#cv2.imshow('edge', edge)

plt.show()
cv2.waitKey(0)

cv2.resize()