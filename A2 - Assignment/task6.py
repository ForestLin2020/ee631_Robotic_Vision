
import numpy as np
import cv2

# load intrinsic and distortion parameters
mtx = np.load('intrinsic  parameters.npy')
dist = np.load('distortion parameters.npy')

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# absdiff camera and then take a picture
cap = cv2.VideoCapture(0)

while True:
    ret, frame1 = cap.read()
    h, w = frame1.shape[:2]

    # Task3 and Task 6, when both shape their images and frame1.
    # windows python 3.7 need to +1, because img.shape change the size of image
    # ask why this happen on windows python but not on linux python
    h += 1
    w += 1

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    # undistort
    dst = cv2.undistort(frame1, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    # absdiff camera
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    cv2.imshow('Differet Detection', diff)
    key = cv2.waitKey(1)

    # Take a distortion picture
    if key % 256 == 27:                # ESC pressed to exit
        print("Escape hit, closing...")
        break

    elif key % 256 == 32:            # SPACE pressed to get an image from camera
        cv2.imwrite('Task6_absdiff.jpg', diff)
        print('Real-Time-absdiff-Task6.jpg is written!')
        break


cv2.destroyAllWindows()