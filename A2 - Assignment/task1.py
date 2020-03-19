import numpy as np
import cv2

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*10,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# when the picture is in "different" fold with task1.py.   
image = cv2.imread('AR1.jpg')
# when the picture is in "same" fold with task1.py.
# image = cv.imread('AR1.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (7,10), None)
# print('corners = ',corners)
# If found, add object points, image points (after refining them)
# print(corners)
if ret == True:
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    # print('corners2 = ',corners2)
    imgpoints.append(corners)

    # Draw and display the corners
    cv2.drawChessboardCorners(image, (7,10), corners2, ret)
    cv2.imshow('image', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()