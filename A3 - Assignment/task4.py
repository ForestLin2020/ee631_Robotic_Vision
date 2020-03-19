import cv2
import numpy as np
import matplotlib.pyplot as plt


# intrinsic and distortion parameter from task1
camera_matrix_l = np.load('L_intrinsic  parameters.npy')
camera_matrix_r = np.load('R_intrinsic  parameters.npy')
dist_coeffs_l = np.load('L_distortion parameters.npy')
dist_coeffs_r = np.load('R_distortion parameters.npy')

# rotation_matrix and translation_vector from task2
rotation_matrix = np.load('rotation_matrix.npy')
translation_vector = np. load('translation_vector.npy')

# Using stereoRectify function to get R1, R2 and P1, P2 and then using then to remap an image
def compute_stereo_rectification_maps(image_name, camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, rotation_matrix, translation_vector):
    image = cv2.imread(image_name)
    h, w = image.shape[:2]
    imgSize = (w, h)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(camera_matrix_l, dist_coeffs_l,
                                                      camera_matrix_r, dist_coeffs_r,
                                                      imgSize,
                                                      rotation_matrix, translation_vector)

    np.save('R1_3x3_rectification_transform.npy', R1)
    np.save('R2_3x3_rectification_transform.npy', R2)
    np.save('P1_3x4_projection_matrix.npy', P1)
    np.save('P2_3x4_projection_matrix.npy', P2)
    np.save('Q_4x4 disparity_to_depth_mapping_matrix.npy', Q)

    map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix_l,
                                               dist_coeffs_l,
                                               R1, P1, imgSize, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix_r,
                                               dist_coeffs_r,
                                               R2, P2, imgSize, cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, roi1, roi2


def RemapPicutre(imagename, camera_matrix, dist_coeffs, map_x, map_y, roi):
    img = cv2.imread(imagename)
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    # remap L
    remap = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    # crop the image

    imagename = 'Rectify' + imagename
    cv2.imwrite(imagename, remap)
    return remap

def Different(imagename_ori, remap):
    # Get picture
    img_ori = cv2.imread(imagename_ori)

    # Compare Original one and Distortion one
    gray1 = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(remap, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)

    cv2.imwrite('Diff' + imagename_ori, diff)
    cv2.imshow('Differet Detection' + imagename_ori, diff)
    return

# The image only for getting the image size
map1x, map1y, map2x, map2y, roi1, roi2 = compute_stereo_rectification_maps('stereo_L11.png', camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, rotation_matrix, translation_vector)

# remap images
remap_L = RemapPicutre('stereo_L11.png', camera_matrix_l, dist_coeffs_l, map1x, map1y, roi1)
remap_R = RemapPicutre('stereo_R11.png', camera_matrix_r, dist_coeffs_r, map2x, map2y, roi2)

# compare the difference
Different('stereo_L11.png', remap_L)
Different('stereo_R11.png', remap_R)

# draw multiple check lines for the result of rectified images
for y in range(20):
    cv2.line(remap_L, (0, y*32), (640, y*32), (0, 0, 255), 1)
    cv2.line(remap_R, (0, y*32), (640, y*32), (0, 0, 255), 1)

cv2.imshow('Rectify_L', remap_L)
cv2.imshow('Rectify_R', remap_R)

cv2.waitKey(0)
cv2.destroyAllWindows()