import numpy as np
import cv2


picture = cv2.imread('detection1.jpg', 0)
image = cv2.imread('reference.jpg', 0) # (756,495)
print('image:', image.shape)

# method of feature points
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create()

# using ORB as my match method
pic_kp, pic_des = orb.detectAndCompute(picture, None)
img_kp, img_des = orb.detectAndCompute(image, None)
# pic_kp = cv2.drawKeypoints(picture, pic_kp, None)
# img_kp = cv2.drawKeypoints(image, img_kp, None)
# cv2.imshow('matching_result',picture)
# cv2.imshow('matching_result',image)
#
# cv2.waitKey(0)


# Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(pic_des, img_des)
# sort the distance from lower to higher
matches = sorted(matches, key=lambda x:x.distance)

# show how many match in picture and image
# the smaller distance the better is matched
print(len(matches))
for m in matches:
    print(m.distance)

matching_result = cv2.drawMatches(image,img_kp,picture,pic_kp,matches[:10],None,flags=2)


cv2.imshow('Feature Matching',matching_result)
cv2.imwrite('Task1_Matching_Result.jpg',matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()