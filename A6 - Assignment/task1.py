import numpy as np
import cv2


picture = cv2.imread('picture.jpg', 0)
image = cv2.imread('image.jpg', 0) # (756,495)
target = cv2.imread('target.jpg',0) #(794,493)
print('image:', image.shape)
print('target:', target.shape)
# method: resized = cv2.resize(img, dim=(width,hight), interpolation = cv2.INTER_AREA)
target.ravel()
width = int(target.shape[1] * (495/493))
hight = int(target.shape[0] * (756/794))
target_resized = cv2.resize(target,(width,hight),interpolation = cv2.INTER_AREA)
print('target_resized:', target_resized.shape)
cv2.imshow('ooo',target_resized)


# method of feature points
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create()

pic_kp, pic_des = orb.detectAndCompute(picture, None)
img_kp, img_des = orb.detectAndCompute(image, None)
# pic_kp = cv2.drawKeypoints(picture, pic_kp, None)
# img_kp = cv2.drawKeypoints(image, img_kp, None)

# Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(pic_des, img_des)
# sort the distance from lower to higher
matches = sorted(matches, key=lambda x:x.distance)

# show how many match in picture and image
# the smaller distance the better is matched  >> why??
# print(len(matches))
# for m in matches:
#     print(m.distance)

matching_result = cv2.drawMatches(image,img_kp,picture,pic_kp,matches[:10],None,flags=2)

# cv2.imshow('picture', pic_kp)
# cv2.imshow('image', img_kp)
cv2.imshow('matching_result',matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()