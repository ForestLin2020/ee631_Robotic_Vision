import cv2
import numpy as np
import glob

cam = cv2.VideoCapture(0)
cv2.namedWindow("Real-Time Capture")
img_counter = 1

# Capture images from camera
while True:
    ret, frame = cam.read()
    cv2.imshow("Real-Time Capture", frame)
    if not ret:
        break
    key = cv2.waitKey(1)

    if img_counter == 41:           # Take 40 pictures to calculate intrinsic and distortion
        break

    if key%256 == 27:               # ESC pressed to exit
        print("Escape hit, closing...")
        break

    elif key%256 == 32:             # SPACE pressed to get images from camera
        img_name = "sample_{}.jpg".format(img_counter)  # {}  =   .format()
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
cam.release()

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# read all images match the objp and append image and object points (7*9)
images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,7),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,7), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(50)
# Find the Intrinsic parameters and Distortion parameters
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print('Real-Time intrinsic  parameters = ',mtx)
print('Real-Time distortion parameters = ',dist)

# save parameters
np.save('Real-Time intrinsic  parameters.npy', mtx)
np.save('Real-Time distortion parameters.npy', dist)

cv2.destroyAllWindows()