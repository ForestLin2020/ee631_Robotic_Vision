import numpy as np
import cv2

L_intrinsic = np.load('L_intrinsic  parameters.npy') 
L_distortion = np.load('L_distortion parameters.npy')
R_intrinsic = np.load('R_intrinsic  parameters.npy')
R_distortion = np.load('R_distortion parameters.npy') 

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*10,3), np.float32)				#(x*y)
objp[:,:2] = 3.88 * np.mgrid[0:10,0:7].T.reshape(-1,2)		#(0:y, 0:x)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
L_imgpoints = [] # 2d points in image plane.
R_imgpoints = [] # 2d points in image plane.

# set up the picture's names
head_list = ["stereo_L", "stereo_R"]

picture_number = []
for i in range(0,34):
    picture_number.append(i)

print('picture_number =', picture_number)

# read all images and add object points and images points
for head in head_list:
    print('head = ', head)
    for pic in picture_number:
        filename = head + str(pic) + ".png"
        print('filename = ', filename)
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (10,7),None)

        # If found, add object points, Left and Right image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            if head == "stereo_L":
                objpoints.append(objp)
                L_imgpoints.append(corners2)
                print('L_imgpoints pic =', pic)
            elif head == "stereo_R":
                R_imgpoints.append(corners2)
                print('R_imgpoints pic =', pic)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (10,7), corners2,ret)
            cv2.imshow('image',img)
            cv2.waitKey(50)

# using stereoCalibrate to get Rotation, Translation, Essential, Fundation 
termination_criteria_extrinsics = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
(rms_stereo, stereo_camera_matrix_l, stereo_dist_coeffs_l, stereo_camera_matrix_r, stereo_dist_coeffs_r, R, T, E, F) = \
    cv2.stereoCalibrate(objpoints, L_imgpoints, R_imgpoints, L_intrinsic, L_distortion, R_intrinsic, R_distortion,  gray.shape[::-1], criteria=termination_criteria_extrinsics, flags=cv2.CALIB_FIX_INTRINSIC)

print('R = ', R)
print('T = ', T)
print('E = ', E)
print('F = ', F)

np.save('rotation_matrix.npy', R)
np.save('translation_vector.npy', T)
np.save('essential_matrix.npy', E)
np.save('fundamental_matrix.npy', F)
