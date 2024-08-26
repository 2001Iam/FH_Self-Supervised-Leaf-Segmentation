import cv2 as  cv
import imutils
import numpy as np
img1 = cv.imread('Output_num99/res(1).png',-1)

img2 = cv.imread('input/img1.jpg')
img2 = imutils.resize(img2, height=800, width=600)
img1 = np.expand_dims(img1, axis=-1)
img = img1//2+img2//2
cv.imshow("depth",img1)
cv.imshow("img2",img2)
cv.imshow("img",img)

cv.waitKey()
a=1





"""cv.imshow("imh",img)
cv.waitKey(1)
cv.imshow("img",img)

print(img)"""