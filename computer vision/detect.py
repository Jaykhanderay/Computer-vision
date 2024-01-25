import cv2
import numpy as np

# reading image fro  the computer and taking dimensions
img = cv2.imread('cat.jpg')
rows,cols= img.shape[:2]

#kernel blurring using filter2d
kernel_25 =np.ones((25,25),np.float32) /625.0
output_kernel = cv2.filter2D(img,-1,kernel_25)

#Boxfiltering and blur function blurring

output_blur=cv2.blur(img,(25,25))
output_box=cv2.boxFilter(img,-1,(5,5),normalize=False)

#Gaussian filter
output_gaus=cv2.GaussianBlur(img,(5,5),0)

#median blur (reduction blur)

output_med=cv2.medianBlur(img,5)
cv2.imshow('kernel_blur',output_kernel)
cv2.imshow('Blur() output',output_kernel)
cv2.imshow('Median Blur',output_med)
cv2.imshow('Box filter',output_box)
cv2.imshow('Original',img)
cv2.waitKey(0)