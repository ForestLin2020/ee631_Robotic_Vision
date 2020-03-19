#!/usr/bin/env python3
import cv2
import numpy as np

# ""=字串, 06=07=08=數字 , "06" and "07"= str(number) 
picture_number = ["06","07","08","09"]

# 對於 number (單)來說 在 range 10-40之間(全) 
for number in range(10,41):
	picture_number.append(str(number))

head_list = ["1L", "1R"]
step = 1

for head in head_list:
	filename = head + "05" + ".jpg"
	img1 = cv2.imread(filename)


	for pic in picture_number:
		filename = head + pic + ".jpg"
		print('filename = ',filename)
		img2 = cv2.imread(filename)
		# difference
		diff = cv2.absdiff(img1,img2)
		# Thresholding it
		gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
		ret,thres = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
		# erosion 
		kernel = np.ones((5,5),np.uint8)
		erosion = cv2.erode(thres,kernel,iterations = 1)
		# dilation
		dilation = cv2.dilate(erosion,kernel,iterations = 2)
		# detect circles in the image and find the position x,y,r
		circles = cv2.HoughCircles(dilation, cv2.HOUGH_GRADIENT, 1,1000, 1,100,1,0,25)
		# loop over the (x, y) coordinates and radius of the circles
		for circles_number in circles:
			x, y, r = circles_number[0]
			#using position(x,y,r) to circle the object in the original picture 
			cv2.circle(img2, (x, y), r, (0, 255, 0), 3)
		
		# show the output image
		cv2.imshow('output',img2)
		cv2.waitKey(40)


cv2.destroyAllWindows() # close the window

