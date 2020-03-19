#!/usr/bin/env python3
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

camera_matrix_l = np.load('L_intrinsic  parameters.npy')
camera_matrix_r = np.load('R_intrinsic  parameters.npy')
dist_coeffs_l = np.load('L_distortion parameters.npy')
dist_coeffs_r = np.load('R_distortion parameters.npy')

R1 = np.load('R1_3x3_rectification_transform.npy')
R2 = np.load('R2_3x3_rectification_transform.npy')
P1 = np.load('P1_3x4_projection_matrix.npy')
P2 = np.load('P2_3x4_projection_matrix.npy')
Q = np.load('Q_4x4 disparity_to_depth_mapping_matrix.npy')

cap_L = cv2.VideoCapture('footage_left.avi')
cap_R = cv2.VideoCapture('footage_right.avi')

# first location catch ball image from L and R
# L(x, y) =  (360.0, 98.5)
# R(x, y) =  (277.5, 99.5)

rectanglecenter_Lx = 360
rectanglecenter_Ly = 98
rectanglecenter_Rx = 277
rectanglecenter_Ry = 99
extension = 68

start_point_L = (292, 30)
end_point_L = (428, 166)
start_point_R = (209, 31)
end_point_R = (345, 167)
# for writing image sequence number
count = 1

ballcenter2d_L = np.zeros((1,1,2))
ballcenter2d_R = np.zeros((1,1,2))

ballcenter3d_L = np.zeros((1,1,3))
ballcenter3d_R = np.zeros((1,1,3))

world_point_x = []
world_point_y = []
world_point_z = []




while True:
	# get the video frame
	ret_L, frame_L1 = cap_L.read()
	ret_R, frame_R1 = cap_R.read()
	# cv2.rectangle(frame, start_point, end_point, color, thickness)
	cv2.rectangle(frame_L1, start_point_L, end_point_L, (255, 0, 0), 2)
	cv2.rectangle(frame_R1, start_point_R, end_point_R, (255, 0, 0), 2)
	# Crop
	cropL1 = frame_L1[rectanglecenter_Ly - extension:rectanglecenter_Ly + extension, rectanglecenter_Lx - extension:rectanglecenter_Lx + extension]
	cropR1 = frame_R1[rectanglecenter_Ry - extension:rectanglecenter_Ry + extension, rectanglecenter_Rx - extension:rectanglecenter_Rx + extension]
	# gray
	gray_L = cv2.cvtColor(cropL1, cv2.COLOR_BGR2GRAY)
	gray_R = cv2.cvtColor(cropR1, cv2.COLOR_BGR2GRAY)
	# threshold
	ret, thres_L = cv2.threshold(gray_L,50,255,cv2.THRESH_BINARY)
	ret, thres_R = cv2.threshold(gray_R,50,255,cv2.THRESH_BINARY)
	# erosion
	kernel = np.ones((5,5),np.uint8)
	erosion_L = cv2.erode(thres_L, kernel,iterations = 1)
	erosion_R = cv2.erode(thres_R, kernel,iterations = 1)
	# dilation
	dilation_L = cv2.dilate(erosion_L,kernel,iterations = 2)
	dilation_R = cv2.dilate(erosion_R,kernel,iterations = 2)

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
		ballcenter2d_L[0,0,0] = x + start_point_L[0]
		ballcenter2d_L[0,0,1] = y + start_point_L[1]
		# print('ballcenter2d_L = ',ballcenter2d_L )
		undistortPoints_L = cv2.undistortPoints(ballcenter2d_L, camera_matrix_l, dist_coeffs_l, R=R1, P=P1)
		ballcenter3d_L[0,0,0] = undistortPoints_L.item(0)
		ballcenter3d_L[0,0,1] = undistortPoints_L.item(1)
		# print('ballcenter3d_L = ', ballcenter3d_L)


		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# only proceed if the radius meets a minimum size
		if radius > 2:
			# draw the circle and center on the frame,
			# then update the list of tracked points
			center = (int(x)+rectanglecenter_Lx - extension, int(y)+rectanglecenter_Ly - extension)
			cv2.circle(frame_L1, center, int(radius), (0, 255, 255), 2)
			cv2.circle(frame_L1, center, 5, (0, 0, 255), -1)

	# only proceed if at least one contour in the "Right" side was found
	if len(contours_R) > 0:

		# waiting instruction when detect the ball
		# set this on the right side is because right side always get the signal slower than right side
		# it means when you get right side point(Rx,Ry), there is a left side point(Lx,Ly) for sure.
		key = cv2.waitKey(0) & 0xFF 		# cv2.waitKey(0) & 0xFF : 0 means screen will wait and stop for your next instruction from keyboard
		if key == ord('w'):  				# write the image when you wait
			img_name = "task2_frames_{}.jpg".format(count)
			cv2.imwrite(img_name, combine_frame)
			print("{} written!".format(img_name))

		c = max(contours_R, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		print('R(x, y) = ', (x, y))
		ballcenter2d_R[0, 0, 0] = x + start_point_R[0]
		ballcenter2d_R[0, 0, 1] = y + start_point_R[1]

		# print('ballcenter2d_R = ', ballcenter2d_R)
		undistortPoints_R = cv2.undistortPoints(ballcenter2d_R, camera_matrix_r, dist_coeffs_r, R=R2, P=P2)
		ballcenter3d_R[0, 0, 0] = undistortPoints_R.item(0)
		ballcenter3d_R[0, 0, 1] = undistortPoints_R.item(1)
		# print('ballcenter3d_R = ', ballcenter3d_R)
		# dealing with disparity
		disparity = ballcenter3d_L[0, 0, 0] - ballcenter3d_R[0, 0, 0]
		ballcenter3d_L[0, 0, 2] = disparity
		ballcenter3d_R[0, 0, 2] = disparity
		print('ballcenter3d_L = ',ballcenter3d_L)
		print('ballcenter3d_R = ',ballcenter3d_R)
		ballcenter3d_L_perspectiveTransform = cv2.perspectiveTransform(ballcenter3d_L, Q)
		print('ballcenter3d_L_perspectiveTransform = ',ballcenter3d_L_perspectiveTransform)

		catcherpoints = ballcenter3d_L_perspectiveTransform - np.array([11.5,29.5,21.5])
		print('catcherpoints = ',catcherpoints)


		world_point_x.append(catcherpoints[0,0,0])
		world_point_y.append(catcherpoints[0,0,1])
		world_point_z.append(catcherpoints[0,0,2])

		print('world_point_x = ', world_point_x)
		print('world_point_y = ', world_point_y)
		print('world_point_z = ', world_point_z)

		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# only proceed if the radius meets a minimum size
		if radius > 2:
			# draw the circle and center on the frame,
			# then update the list of tracked points
			center = (int(x)+rectanglecenter_Rx - extension, int(y)+rectanglecenter_Ry - extension)
			cv2.circle(frame_R1, center, int(radius), (0, 255, 255), 2)
			cv2.circle(frame_R1, center, 5, (0, 0, 255), -1)
			count +=1


	# show the output image
	combine_frame = np.hstack((frame_L1,frame_R1))
	cv2.imshow('footage_combine', combine_frame)

	key = cv2.waitKey(1) & 0xff # cv2.waitKey(1) & 0xFF : 1 means screen will "not" stop, but still wait for your next instruction from keyboard
	if key == ord('q'):
		cap_L.release()

		image = cv2.imread('stereo_R25.png')
		cv2.imshow('ddd', image)
		key = cv2.waitKey(0)
	# plot the x-z graphs
	if key == ord('x'):
		x = np.array(world_point_x)
		z = np.array(world_point_z)
		world_point_x =[]
		world_point_z =[]
		print('world_point_x = ',world_point_x)
		print('world_point_z = ',world_point_z)

		x_coeff = np.polyfit(z, x, 3)
		p = np.poly1d(x_coeff)
		zp = np.linspace(450, 0, 100)
		plt.xlabel('Z position')
		plt.ylabel('X position')
		plt.plot(z, x, '.')
		plt.plot(zp, p(zp), '-')
		plt.ylim(0, -70)
		plt.show()


	# plot the y-z graphs
	if key == ord('y'):
		y = np.array(world_point_y)
		z = np.array(world_point_z)
		world_point_y = []
		world_point_z = []
		print('world_point_y = ',world_point_y)
		print('world_point_z = ',world_point_z)

		# z = np.polyfit(x, y, 3)
		y_coeff = np.polyfit(z, y, 3)
		p = np.poly1d(y_coeff)
		zp = np.linspace(450, 0, 100)
		plt.xlabel('Z position')
		plt.ylabel('Y position')
		plt.plot(z, y, '.')
		plt.plot(zp, p(zp), '-')
		plt.ylim(0, -70)
		plt.show()

cv2.destroyAllWindows() 	# close the window

