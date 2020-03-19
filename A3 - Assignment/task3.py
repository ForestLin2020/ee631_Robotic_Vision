import numpy as np
import cv2

# intrinsic and distortion parameter from task1
camera_matrix_l = np.load('L_intrinsic  parameters.npy')
camera_matrix_r = np.load('R_intrinsic  parameters.npy')
dist_coeffs_l = np.load('L_distortion parameters.npy')
dist_coeffs_r = np.load('R_distortion parameters.npy')

# fundamental_matrix from task2
fundamental_matrix = np.load('fundamental_matrix.npy')

# Using Undistort function to undistort images
def undistortion(name, camera_matrix, dist_coeffs):
    
    # Get picture
    img = cv2.imread(name)
    
    # Refine the camera matrix
    h, w = img.shape[:2]
    
    # windows python 3.7 need to +1, because img.shape change the size of image
    # ask why this happen on windows python but not on linux python
    h += 1
    w += 1
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0, (w, h))
    
    # undistort
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite('Undistortion' + name, dst)
    return dst

# find object and images points from chessboard pattern and then find corners.
def getCorners(image_name, chessboard_size):
    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:, :2] = 3.88*np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    image = cv2.imread(image_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (chessboard_size[1], chessboard_size[0]), None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # print('corners2 = ',corners2)
        imgpoints.append(corners)
    return imgpoints

# draw the provided points on the image
def drawPoints(img, pts, colors):
    for pt, color in zip(pts, colors):
        cv2.circle(img, tuple(pt[0]), 5, color, -1)


# draw the provided lines on the image
def drawLines(img, lines, colors):
    _, c, _ = img.shape
    for r, color in zip(lines, colors):
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cv2.line(img, (x0, y0), (x1, y1), color, 1)

# undistort images
dst_L = undistortion('stereo_L11.png', camera_matrix_l, dist_coeffs_l)
dst_R = undistortion('stereo_R11.png', camera_matrix_r, dist_coeffs_r)
gray_L = cv2.cvtColor(dst_L, cv2.COLOR_BGR2GRAY)
gray_R = cv2.cvtColor(dst_R, cv2.COLOR_BGR2GRAY)

# use get corners to get the new image locations of the checcboard corners
imgpoints_L = getCorners('stereo_L11.png', (7,10))
imgpoints_R = getCorners('stereo_R11.png', (7,10))

# get 3 image points of interest from each image and draw them
ptsL = np.asarray([imgpoints_L[-1][2], imgpoints_L[0][-4], imgpoints_L[0][7]]) 
ptsR = np.asarray([imgpoints_R[-1][7], imgpoints_R[0][15], imgpoints_R[0][25]])
drawPoints(dst_L, ptsL, (250, 0, 0))
drawPoints(dst_R, ptsR, (150, 255, 200))

# find epilines corresponding to points in right image and draw them on the left image
epilinesR = cv2.computeCorrespondEpilines(ptsR.reshape(-1, 1, 2), 2, fundamental_matrix)
epilinesR = epilinesR.reshape(-1, 3)
drawLines(dst_L, epilinesR, (150, 255, 200))

# find epilines corresponding to points in left image and draw them on the right image
epilinesL = cv2.computeCorrespondEpilines(ptsL.reshape(-1, 1, 2), 1, fundamental_matrix)
epilinesL = epilinesL.reshape(-1, 3)
drawLines(dst_R, epilinesL, (255, 100, 255))

cv2.imshow('CorrespondEpilines_L',dst_L)
cv2.imshow('CorrespondEpilines_R',dst_R)

cv2.waitKey(0)
cv2.destroyAllWindows()

