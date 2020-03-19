import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*10,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


# read all images match the objp and append image and object points (7*10)
images = glob.glob('*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (10,7),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (10,7), corners2,ret)
        cv2.imshow('image',img)
        cv2.waitKey(50)

# Find the Intrinsic parameters and Distortion parameters
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print('intrinsic  parameters = ',mtx)
print('distortion parameters = ',dist)

# Calculate the actual focal length
fsx = mtx[0,0]
print('fsx = ', fsx)
focal_length = fsx * 7.4e-6 * 1000
print('focal_length in mm = {} mm'.format(focal_length))

# save parameters
np.save('intrinsic  parameters.npy', mtx)
np.save('distortion parameters.npy', dist)

cv2.destroyAllWindows()