import numpy as np
import cv2

camera_matrix_l = np.load('L_intrinsic  parameters.npy')
camera_matrix_r = np.load('R_intrinsic  parameters.npy')
dist_coeffs_l = np.load('L_distortion parameters.npy')
dist_coeffs_r = np.load('R_distortion parameters.npy')

R1 = np.load('R1_3x3_rectification_transform.npy')
R2 = np.load('R2_3x3_rectification_transform.npy')
P1 = np.load('P1_3x4_projection_matrix.npy')
P2 = np.load('P2_3x4_projection_matrix.npy')
Q = np.load('Q_4x4 disparity_to_depth_mapping_matrix.npy')

def getcorners(imagename):
	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((7*10,3), np.float32)
	objp[:,:2] = 3.88*np.mgrid[0:10,0:7].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	# when the picture is in "different" fold with task1.py.
	image = cv2.imread(imagename)
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
		imgpoints.append(corners)

		# Draw and display the corners
		# cv2.drawChessboardCorners(image, (2,2), corners2, ret)
		# cv2.imshow(imagename, image)
		# cv2.waitKey(0)
	return corners2

def drawPoints(img, pts, colors):
    for pt, color in zip(pts, colors):
        cv2.circle(img, tuple(pt[0]), 5, color, -1)

# get the image corners
corners_L = getcorners('stereo_L7.png')
corners_R = getcorners('stereo_R7.png')

image_L = cv2.imread('stereo_L7.png')
image_R = cv2.imread('stereo_R7.png')

# getting the four outermost points to a new array
fourPoints_L = np.array([corners_L[0], corners_L[6], corners_L[63],corners_L[69]])
fourPoints_R = np.array([corners_R[0], corners_R[6], corners_R[63],corners_R[69]])

cv2.drawChessboardCorners(image_L, (2,2), fourPoints_L, True)
cv2.drawChessboardCorners(image_R, (2,2), fourPoints_R, True)

combine_image = np.hstack((image_L,image_R))
cv2.imshow('combine_image_L_and_R.png', combine_image)
key = cv2.waitKey(0) & 0xFF 		# cv2.waitKey(0) & 0xFF : 0 means screen will wait and stop for your next instruction from keyboard
if key == ord('w'):  				# write the image when you wait
	cv2.imwrite('combine_image_L_and_R.png', combine_image)
	print('image is ritten!')


print('fourPoints_L',fourPoints_L)
undistortPoints_L = cv2.undistortPoints(fourPoints_L, camera_matrix_l, dist_coeffs_l, R=R1, P=P1)
undistortPoints_R = cv2.undistortPoints(fourPoints_R, camera_matrix_r, dist_coeffs_r, R=R2, P=P2)

# print('undistortPoints_L =',undistortPoints_L)
# print('undistortPoints_R =',undistortPoints_R)

disparity = np.zeros([1,4])
disparity[0,0] = undistortPoints_L.item(0) - undistortPoints_R.item(0)
disparity[0,1] = undistortPoints_L.item(2) - undistortPoints_R.item(2)
disparity[0,2] = undistortPoints_L.item(4) - undistortPoints_R.item(4)
disparity[0,3] = undistortPoints_L.item(6) - undistortPoints_R.item(6)

print('disparity = ', disparity)

point_3d_L = np.zeros((4,1,3))
for i in range(4):
	point_3d_L[i, 0, 2] = disparity[0, i]
	for j in range(2):
		point_3d_L[i, 0, j] = undistortPoints_L[i, 0, j]

point_3d_R = np.zeros((4,1,3))
for i in range(4):
	point_3d_R[i, 0, 2] = disparity[0, i]
	for j in range(2):
		point_3d_R[i, 0, j] = undistortPoints_R[i, 0, j]

# point_3d_L[0,0,0] = undistortPoints_L[0,0,0]
# point_3d_L[0,0,1] = undistortPoints_L[0,0,1]
# point_3d_L[0,0,2] = disparity[0,0]
#
# point_3d_L[1,0,0] = undistortPoints_L[1,0,0]
# point_3d_L[1,0,1] = undistortPoints_L[1,0,1]
# point_3d_L[1,0,2] = disparity[0,1]
#
# point_3d_L[2,0,0] = undistortPoints_L[2,0,0]
# point_3d_L[2,0,1] = undistortPoints_L[2,0,1]
# point_3d_L[2,0,2] = disparity[0,2]
#
# point_3d_L[3,0,0] = undistortPoints_L[3,0,0]
# point_3d_L[3,0,1] = undistortPoints_L[3,0,1]
# point_3d_L[3,0,2] = disparity[0,3]

print('point_3d_L = ', point_3d_L)
print('point_3d_R = ', point_3d_R)

perspectiveTransform_L = cv2.perspectiveTransform(point_3d_L, Q)
print('perspectiveTransform_L = ', perspectiveTransform_L)

perspectiveTransform_R = cv2.perspectiveTransform(point_3d_R, Q)
print('perspectiveTransform_R = ', perspectiveTransform_R)

cv2.destroyAllWindows()
