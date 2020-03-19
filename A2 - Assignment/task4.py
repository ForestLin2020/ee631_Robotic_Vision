import cv2
import numpy as np


# Load previously saved data base on task2
mtx = np.load('intrinsic  parameters.npy')
dist = np.load('distortion parameters.npy')

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Get picture and Data Points
image = cv2.imread('Object with Corners.jpg')
data_points = open('Data Points.txt')

for line in data_points:    # deal with data line by line
    data = [float(x) for x in line.split()]     # separate every factor by detecting block space
    if len(data) == 2:          # two factors for image points
        imgpoints.append(data)
    elif len(data) == 3:        # three factors for object points
        objpoints.append(data)
    else:
        exit()

# Change to array of float32
imgpoints = np.asarray(imgpoints, dtype=np.float32)
objpoints = np.asarray(objpoints, dtype=np.float32)

# Find rotation and translation vectors.
rvecs, tvecs = cv2.solvePnP(objpoints, imgpoints, mtx, dist)[1:3]

print('rvecs = ', rvecs)
print('tvecs = ', tvecs)

# Find rotation and translation matrices.
rmat = cv2.Rodrigues(rvecs)[0]
tmat = cv2.Rodrigues(tvecs)[0]

print('rmat = ', rmat)
print('tmat = ', tmat)


cv2.destroyAllWindows()





