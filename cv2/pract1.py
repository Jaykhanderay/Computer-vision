import cv2 
import numpy as np
img=cv2.imread(r'C:\Users\RCPIT\Desktop\flower3.jpg')
b,g,r=cv2.split(img)
zeros_ch=np.zeros(img.shape[0:2],dtype="uint8")

#blue image
blue_img =cv2.merge([b,zeros_ch,zeros_ch])
cv2.imshow('Blue Image',blue_img)
cv2.waitkey()

#Green img 

green_img=cv2.merge([zeros_ch,g,zeros_ch])
cv2.imshow('Green Image',green_img)
cv2.waitKey()

#Red image
red_img -= cv2.merge([zeros_ch,zeros_ch,r])
cv2.imshow('Red Image',red_img)
cv2.waitKey()