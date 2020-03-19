#!/usr/bin/env python3

from skimage.measure import compare_ssim
import cv2 
import numpy as np




cap = cv2.VideoCapture(0)
detection = "1"

while True :

	# Run the Detection function
	if detection == "1": # Original
		# Capture frame-by-frame
		ret,frame = cap.read()
		cv2.imshow("Original",frame)
		key = cv2.waitKey(1) & 0xFF

	elif detection == "2": # thresholding
		ret,frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		thres = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)[1] #[1]??????
		cv2.imshow("thresholding Detection",thres)
		key = cv2.waitKey(1) & 0xFF

	elif detection == "3": # Canny Edge 
		ret,frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray,100,200)
		cv2.imshow("Edges Detection",edges)
		key = cv2.waitKey(1) & 0xFF

	elif detection == "4": # Corner 
		ret,frame = cap.read()
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		gray = np.float32(gray)
		dst = cv2.cornerHarris(gray,2,3,0.04)
		#result is dilated for marking the corners, not important
		dst = cv2.dilate(dst,None)
		# Threshold for an optimal value, it may vary depending on the image.
		frame[dst>0.01*dst.max()]=[0,0,255]
		cv2.imshow('Corner Detection',frame)
		key = cv2.waitKey(1) & 0xFF
	elif detection == "5": # Lines 
		ret,frame = cap.read()
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray,50,150,apertureSize = 3)
		minLineLength = 100
		maxLineGap = 10
		lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
		for x1,y1,x2,y2 in lines[0]:
			cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
		cv2.imshow('Lines Detection',frame)
		key = cv2.waitKey(1) & 0xFF

	elif detection == "6": # Difference 
		ret,frame1 = cap.read()
		gray1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
		ret,frame2 = cap.read()
		gray2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
		diff = cv2.absdiff(gray1,gray2)
		cv2.imshow('Differet Detection',diff)
		key = cv2.waitKey(1) & 0xFF


	# Set the Detection function
	if key == ord('q'):
		break
		cap.release()
	elif key == ord('1'):
		detection = "1"
		cv2.destroyAllWindows()
	elif key == ord('2'):
		detection = "2"
		cv2.destroyAllWindows()
	elif key == ord('3'):
		detection = "3"
		cv2.destroyAllWindows()
	elif key == ord('4'):
		detection = "4"
		cv2.destroyAllWindows()
	elif key == ord('5'):
		detection = "5"
		cv2.destroyAllWindows()
	elif key == ord('6'):
		detection = "6"
		cv2.destroyAllWindows()


