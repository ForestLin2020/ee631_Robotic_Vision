import numpy as np
import cv2

cap = cv2.VideoCapture('livingroom.mp4')
# counting frame number
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('frame_count = ', frame_count)
frame_number = 0

while True:
    # Take first frame and find corners in it
    ret, old_frame = cap.read()

    ####### skip frame >> frame_number += 10 or 20 #######
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    # print('Position:', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    point0 = cv2.goodFeaturesToTrack(old_gray,
                                 mask=None,
                                 maxCorners=100,
                                 qualityLevel=0.1,
                                 minDistance=10,
                                 blockSize=7)

    ret, new_frame = cap.read()
    new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    # Set the Pyramid level = maxLevel
    # cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts[, nextPts[, status[, err[, winSize[, maxLevel[, criteria[, flags[, minEigThreshold]]]]]]]]) → nextPts, status, err
    point_m, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                           new_gray,
                                           point0,
                                           None,
                                           winSize=(15,15),
                                           maxLevel=4,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Select good points
    good_new = point_m[st==1]
    good_old = point0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        # print('good_new', good_new)

        # .ravel = .reshape(-1, order=order)
        a,b = new.ravel()
        c,d = old.ravel()

        # image = cv2.line(image, start_point, end_point, color, thickness)
        new_frame = cv2.line(new_frame, (a,b),(c,d), (0,0,255), 2)
        # image = cv2.circle(image, center_coordinates, radius, color, thickness)
        new_frame = cv2.circle(new_frame,(a,b),3,(0,255,0),3)

        # cv2.waitKey(0)

    #image = cv2.putText(image, 'OpenCV', org, font, fontScale, color, thickness, cv2.LINE_AA)
    # cv2.putText(new_frame,'Pyramid level:0  Skipping Frame:10', (0, 1080), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    new_frame = cv2.putText(new_frame, 'Pyramid Level:4  Skipped Frames:10', (400, 700), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 4, cv2.LINE_AA)

    cv2.imshow('frame',new_frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

    # skip frame number
    frame_number += 10


cv2.destroyAllWindows()
cap.release()