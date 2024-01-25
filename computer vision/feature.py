import cv2
import numpy as np

img = cv2.imread("cat.jpg",cv2.IMREAD_GRAYSCALE)

sift=cv2.xfeatures2d.SIFT_create()
#surf=cv2.xfeatures2d.SURF_create()

#keypoints,descriptors = surf.detectAndCompute(img,None)

kp=sift.detect(img,None)

img=cv2.drawKeypoints(img,kp,None)

cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()