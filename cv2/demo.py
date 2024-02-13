# import cv2
# import numpy as np

# # read the input color image
# img = cv2.imread('flower3.jpg')

# # split the Blue, Green and Red color channels
# blue,green,red = cv2.split(img)

# # define channel having all zeros
# zeros = np.zeros(blue.shape, np.uint8)

# # merge zeros to make BGR image
# blueBGR = cv2.merge([blue,zeros,zeros])
# greenBGR = cv2.merge([zeros,green,zeros])
# redBGR = cv2.merge([zeros,zeros,red])

# # display the three Blue, Green, and Red channels as BGR image
# cv2.imshow('Blue Channel', blueBGR)
# cv2.waitKey(0)
# cv2.imshow('Green Channel', greenBGR)
# cv2.waitKey(0)
# cv2.imshow('Red Channel', redBGR)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# Blurring the image using the Gaussian blur,median Blur and Bilateral blur
 

# importing libraries 
# import cv2 
# import numpy as np 

# image = cv2.imread('flower3.jpg') 

# cv2.imshow('Original Image', image) 
# cv2.waitKey(0) 

# # Gaussian Blur 
# Gaussian = cv2.GaussianBlur(image, (7, 7), 0) 
# cv2.imshow('Gaussian Blurring', Gaussian) 
# cv2.waitKey(0) 

# # Median Blur 
# median = cv2.medianBlur(image, 5) 
# cv2.imshow('Median Blurring', median) 
# cv2.waitKey(0) 


# # Bilateral Blur 
# bilateral = cv2.bilateralFilter(image, 9, 75, 75) 
# cv2.imshow('Bilateral Blurring', bilateral) 
# cv2.waitKey(0) 
# cv2.destroyAllWindows()


# Image resizing using opencv

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# image = cv2.imread('flower3.jpg')
# # Loading the image

# half = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1)
# bigger = cv2.resize(image, (1050, 1610))

# stretch_near = cv2.resize(image, (780, 540), 
# 			interpolation = cv2.INTER_LINEAR)


# Titles =["Original", "Half", "Bigger", "Interpolation Nearest"]
# images =[image, half, bigger, stretch_near]
# count = 4

# for i in range(count):
# 	plt.subplot(2, 2, i + 1)
# 	plt.title(Titles[i])
# 	plt.imshow(images[i])

# plt.show()

# Image Detection using canny
# import cv2
# import numpy as np

# image =cv2.imread('flower3.jpg')
# image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# cv2.resize(image, (2150, 1610))



# #performing the edge detection 

# gradients_sobelx=cv2.Sobel(image,-1,1,0)
# gradients_sobely=cv2.Sobel(image,-1,0,1)
# gradients_sobelxy=cv2.addWeighted(gradients_sobelx,0.5,gradients_sobely,0.5,0)

# #laplacian

# gradient_laplacian=cv2.Laplacian(image,-1)

# canny_ouput=cv2.Canny(image,80,150)

# cv2.imshow('Sobel x',gradients_sobelx)
# cv2.imshow('Sobel y',gradients_sobely)
# cv2.imshow('Sobel X+y',gradients_sobelxy)
# cv2.imshow('laplacian',gradient_laplacian)
# cv2.imshow('canny',canny_ouput)
# cv2.waitKey()

# import cv2
# import numpy as np

# img = cv2.imread("flower3.jpg",cv2.IMREAD_GRAYSCALE)

# sift=cv2.xfeatures2d.SIFT_create()
# surf=cv2.xfeatures2d.SURF_create()

# keypoints,descriptors = surf.detectAndCompute(img,None)

# #kp=sift.detect(img,None)

# img=cv2.drawKeypoints(img,kp,None)

# cv2.imshow("img",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# img = cv2.imread('flower3.jpg')
# rows,cols,_ = img.shape

# M_left = np.float32([[1,0,-50],[0,1,0]])
# M_right = np.float32([[1,0,50],[0,1,0]])
# M_top = np.float32([[1,0,0],[0,1,50]])
# M_bottom = np.float32([[1,0,0],[0,1,-50]])

# img_left = cv2.warpAffine(img,M_left,(cols,rows))
# img_right = cv2.warpAffine(img,M_right,(cols,rows))
# img_top = cv2.warpAffine(img,M_top,(cols,rows))
# img_bottom = cv2.warpAffine(img,M_bottom,(cols,rows))

# plt.subplot(221), plt.imshow(img_left), plt.title('Left')
# plt.subplot(222), plt.imshow(img_right), plt.title('Right')
# plt.subplot(223), plt.imshow(img_top), plt.title('Top')
# plt.subplot(224), plt.imshow(img_bottom), plt.title('Bottom')
# plt.show()

import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
image = cv2.imread('flower3.jpg') 
#converting image to Gray scale 
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#plotting the grayscale image
plt.imshow(gray_image) 
#converting image to HSV format
hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
#plotting the HSV image
plt.imshow(hsv_image)