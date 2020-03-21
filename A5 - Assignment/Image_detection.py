import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

while True:

    template = cv2.imread('elephant_75.jpg', 0)
    w, h = template.shape[::-1]
    cv2.imshow('template', template)

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    ### method : cv2.TM_SQDIFF ###
    # res = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF)
    # cv2.imshow('res = ', res)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # top_left = min_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv2.rectangle(frame, top_left, bottom_right, 255, 1)

    ###  method : cv2.TM_CCOEFF_NORMED  ###
    res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.6
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    cv2.imshow('Test', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# img_rgb = cv2.imread('mario.png')
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# template = cv2.imread('mario_coin.png',0)
# w, h = template.shape[::-1]
#
# res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
# threshold = 0.8
# loc = np.where( res >= threshold)
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
#
# cv2.imwrite('res.png',img_rgb)

cv2.release
cv2.distroyAllwindows