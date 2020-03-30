import numpy as np
import cv2

image = cv2. imread('image.jpg',0)
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
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_grayframe, des_grayframe = sift.detectAndCompute(grayframe,None)
    # frame = cv2.drawKeypoints(frame, kp_grayframe, None)

    matches = flann.knnMatch(des_image,des_grayframe,k=2)
    good_points = []
    for m, n in matches:
        if m.distance < 0.5*n.distance:         # !!! ask more detail about how distance charge the match level
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
        distort = cv2.perspectiveTransform(points, matrix)

        homography = cv2.polylines(frame, [np.int32(distort)], True, (255, 0, 0), 3)

        cv2.imshow('Homography',homography)
    else:
        cv2.imshow('Homography',grayframe)







    # cv2.imshow('image', image)
    # cv2.imshow('frame', frame)
    cv2.imshow('matches_result', matches_result)



    key = cv2.waitKey(1)
    if key == 27:
        break


cv2.release()
cv2.destroyAllWindows()