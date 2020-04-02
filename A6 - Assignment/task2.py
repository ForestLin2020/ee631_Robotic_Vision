import numpy as np
import cv2


image = cv2.imread('image.jpg', 0) # (756,495)
target = cv2.imread('target.jpg') #(794,493)
graytarget = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
flag = cv2.imread('flage.jpg')

print('image:', image.shape)
print('target:', graytarget.shape)
# method: resized = cv2.resize(img, dim=(width,hight), interpolation = cv2.INTER_AREA)
graytarget.ravel()
width = int(graytarget.shape[1] * (495/493))
hight = int(graytarget.shape[0] * (756/794))
target_resized = cv2.resize(graytarget,(width,hight),interpolation = cv2.INTER_AREA)
print('target_resized:', target_resized.shape)

cap = cv2.VideoCapture(0)
sift = cv2.xfeatures2d.SIFT_create()
kp_image, des_image = sift.detectAndCompute(image,None)
# image = cv2.drawKeypoints(image, kp_image, None)

# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    ret, frame = cap.read()
    print('frame.shape', frame.shape)

    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_grayframe, des_grayframe = sift.detectAndCompute(grayframe,None)
    # frame = cv2.drawKeypoints(frame, kp_grayframe, None)

    matches = flann.knnMatch(des_image,des_grayframe, k=2)
    good_points = []
    for m, n in matches:
        if m.distance < 0.5*n.distance:         
            good_points.append(m)

    # draw matches points
    matches_result = cv2.drawMatches(image,kp_image,frame,kp_grayframe,good_points,None, flags=2)

    # Homography
    if len(good_points) > 10:
        image_points = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        frame_points = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(image_points, frame_points, cv2.RANSAC, 5.0)

        # Perspective transformation
        h, w = image.shape
        points = np.float32([[0,0],[0,h-6],[w-6,h-6],[w-6,0]]).reshape(-1, 1, 2)
        distort_pts = cv2.perspectiveTransform(points, matrix)
        print('distort_pts',distort_pts)
        homography = cv2.polylines(frame, [np.int32(distort_pts)], True, (255, 0, 0), 3)

        # warpPerspective(image, newImage, lastHomography, image.size(), INTER_LINEAR | WARP_INVERSE_MAP, BORDER_TRANSPARENT);
        warp_result = cv2.warpPerspective(target, matrix, (homography.shape[1], homography.shape[0]))

        gray_warp = cv2.cvtColor(warp_result,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray_warp, 10, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        warp_result_centerblack = cv2.bitwise_and(homography, homography, mask=mask)
        combine = cv2.add(warp_result, warp_result_centerblack)

        # cv2.imshow('Homography', homography)
        # cv2.imshow('warp_result', warp_result)
        cv2.imshow('combine', combine)
    else:
        cv2.imshow('Homography',grayframe)

    cv2.imshow('matches_result', matches_result)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key =='s':
        cv2.imwrite('')

cv2.release()
cv2.destroyAllWindows()