#!/usr/bin/env python3
import cv2
import numpy as np
import imutils

cap_L = cv2.VideoCapture('footage_left.avi')
cap_R = cv2.VideoCapture('footage_right.avi')

ret_L, frame_L = cap_L.read()
ret_L, frame_R = cap_L.read()

# first location catch ball image from L and R
# L(x, y) =  (360.0, 98.5)
# R(x, y) =  (277.5, 99.5)

rectanglecenter_Lx = 360
rectanglecenter_Ly = 98

rectanglecenter_Rx = 277
rectanglecenter_Ry = 99

extension = 68

start_point_L = (rectanglecenter_Lx - extension, rectanglecenter_Ly - extension)
end_point_L = (rectanglecenter_Lx + extension, rectanglecenter_Ly + extension)

start_point_R = (rectanglecenter_Rx - extension, rectanglecenter_Ry - extension)
end_point_R = (rectanglecenter_Rx + extension, rectanglecenter_Ry + extension)

# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

count = 1

while True:

	ret_L, frame_L1 = cap_L.read()
	ret_R, frame_R1 = cap_R.read()

	# cv2.rectangle(frame_L2, start_point, end_point, color, thickness)
	cv2.rectangle(frame_L1, start_point_L, end_point_L, color, thickness)
	cv2.rectangle(frame_R1, start_point_R, end_point_R, color, thickness)

	cropL1 = frame_L1[rectanglecenter_Ly - extension:rectanglecenter_Ly + extension, rectanglecenter_Lx - extension:rectanglecenter_Lx + extension]
	cropR1 = frame_R1[rectanglecenter_Ry - extension:rectanglecenter_Ry + extension, rectanglecenter_Rx - extension:rectanglecenter_Rx + extension]


	gray_L = cv2.cvtColor(cropL1, cv2.COLOR_BGR2GRAY)
	gray_R = cv2.cvtColor(cropR1, cv2.COLOR_BGR2GRAY)

	ret, thres_L = cv2.threshold(gray_L,50,255,cv2.THRESH_BINARY)
	ret, thres_R = cv2.threshold(gray_R,50,255,cv2.THRESH_BINARY)

	# erosion
	kernel = np.ones((5,5),np.uint8)
	erosion_L = cv2.erode(thres_L, kernel,iterations = 1)
	erosion_R = cv2.erode(thres_R, kernel,iterations = 1)

	# dilation
	dilation_L = cv2.dilate(erosion_L,kernel,iterations = 2)
	dilation_R = cv2.dilate(erosion_R,kernel,iterations = 2)

	cv2.imshow('cropL1', dilation_L)
	cv2.imshow('cropR2', dilation_R)


	# find contours in the dilation_L and initialize the current
	# (x, y) center of the ball
	contours_L = cv2.findContours(dilation_L.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours_R = cv2.findContours(dilation_R.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	contours_L = imutils.grab_contours(contours_L)
	contours_R = imutils.grab_contours(contours_R)
	center = None

	# only proceed if at least one contour in the "Left" side was found
	if len(contours_L) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(contours_L, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		print('L(x, y) = ',(x, y))
		# First(x,y) for Left's footage is (360,98)

		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# only proceed if the radius meets a minimum size
		if radius > 2:
			# draw the circle and center on the frame,
			# then update the list of tracked points
			cv2.circle(frame_L1, (int(x)+rectanglecenter_Lx - extension, int(y)+rectanglecenter_Ly - extension), int(radius), (0, 255, 255), 2)
			cv2.circle(frame_L1, (int(x)+rectanglecenter_Lx - extension, int(y)+rectanglecenter_Ly - extension), 5, (0, 0, 255), -1)

	# only proceed if at least one contour in the "Right" side was found
	if len(contours_R) > 0:

		# waiting instruction when detect the ball
		# set this on the right side is because right side always get the signal slower than right side
		# it means when you get right side point(Rx,Ry), there is a left side point(Lx,Ly) for sure.
		key = cv2.waitKey(0) & 0xFF
		if key == ord('w'):  # write the image when you wait
			img_name = "task2_frames_{}.jpg".format(count)
			cv2.imwrite(img_name, combine_frame)
			print("{} written!".format(img_name))

		c = max(contours_R, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		print('R(x, y) = ', (x, y))
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# only proceed if the radius meets a minimum size
		if radius > 2:
			# draw the circle and center on the frame,
			# then update the list of tracked points
			cv2.circle(frame_R1, (int(x)+rectanglecenter_Rx - extension, int(y)+rectanglecenter_Ry - extension), int(radius), (0, 255, 255), 2)
			cv2.circle(frame_R1, (int(x)+rectanglecenter_Rx - extension, int(y)+rectanglecenter_Ry - extension), 5, (0, 0, 255), -1)
			count +=1


	# show the output image
	combine_frame = np.hstack((frame_L1,frame_R1))
	cv2.imshow('footage_combine', combine_frame)

	key = cv2.waitKey(1) & 0xff
	if key == ord('q'):
		cap_L.release()


cv2.destroyAllWindows() # close the window