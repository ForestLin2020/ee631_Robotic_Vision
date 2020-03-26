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

    # skip frame >> frame_number += 10 or 20
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    # print('Position:', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_points = cv2.goodFeaturesToTrack(old_gray,
                                 mask=None,
                                 maxCorners=50,
                                 qualityLevel=0.1,
                                 minDistance=10,
                                 blockSize=7)

    ret, new_frame = cap.read()
    new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('new_gray',new_gray)
    new_points = np.zeros_like(old_points)
    i = 0
    for point in old_points:
        point = point.ravel()
        search_window = new_gray[int(point[1]):int(point[1]+50), int(point[0]):int(point[0]+50)]
        template_window = old_gray[int(point[1]):int(point[1]+10), int(point[0]):int(point[0]+10)]
        resmat = cv2.matchTemplate(search_window, template_window, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(resmat)
        point1 = [int(point[0])+max_loc[0],int(point[1])+max_loc[1]]
        new_points[i] = point1
        i += 1

    old_points = np.int32(old_points)
    new_points = np.int32(new_points)
    F, mask = cv2.findFundamentalMat(old_points, new_points, cv2.FM_LMEDS)

    # We select only inlier points
    old_points = old_points[mask.ravel() == 1]
    new_points = new_points[mask.ravel() == 1]


    # draw the tracks
    for i,(new,old) in enumerate(zip(new_points,old_points)):
        # .ravel = .reshape(-1, order=order)
        a,b = new.ravel()
        c,d = old.ravel()

        # image = cv2.line(image, start_point, end_point, color, thickness)
        new_frame = cv2.line(new_frame, (a,b),(c,d), (0,0,255), 2)
        # image = cv2.circle(image, center_coordinates, radius, color, thickness)
        new_frame = cv2.circle(new_frame,(a,b),3,(0,255,0),3)

    # image = cv2.putText(image, 'OpenCV', org, font, fontScale, color, thickness, cv2.LINE_AA)
    new_frame = cv2.putText(new_frame, 'Skipped Frames:0', (400, 700), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 4, cv2.LINE_AA)

    cv2.imshow('frame',new_frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

    # skip frame number
    # frame_number += 10


cv2.destroyAllWindows()
cap.release()