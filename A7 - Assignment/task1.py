import cv2
import numpy as np
import matplotlib.pyplot as plt

# parameters = np.loadtxt('TTI Camera Parameters.txt')
parameters = np.array([[825.0900600547, 0.0000000000, 331.6538103208],
                       [0.0000000000, 824.2672147458, 252.9284287373],
                       [0.0000000000, 0.0000000000, 1.0000000000]])
print('parameters', parameters)
fsx = parameters[0][0]
print('fsx', fsx)
# !!!!!!!!!!!!!!!!! need to check
# guess pixel size(length/pixel): 7.4e-6 (micro/pixel)
focal_length = fsx * 7.4e-6 * 1000  # (micro/pixel) * 1000 = milli >> mm
# {} << .format(name of value)
print('focal length in mm = {} mm'.format(focal_length))

x_prime = 0
x = 0
frame_list = []
Z_list = []
tau_list = []

def distance(x_pixel):

    Z = fsx * 59 / x_pixel
    Z_list.append(Z)
    print('Distance Z in mm = {} mm'.format(Z))
    Z_prime = Z - 15.25
    # The rate of expansion (a)
    a = Z / Z_prime
    print('The rate of expansion = ', a)
    # Time to impact: tau
    tau = a / (a-1)
    print('Time to impact (tau):',tau)

    return Z

######################[ getting the x in pixel]#################################

image = cv2.imread('gas_can.jpg', 0)
cap = cv2.VideoCapture('gascan_0.1s.mp4')
sift = cv2.xfeatures2d.SIFT_create()
kp_image, des_image = sift.detectAndCompute(image,None)
# image = cv2.drawKeypoints(image, kp_image, None)

# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)
frame_number = 0

while True:
    ret, frame = cap.read()

    frame_list.append(frame_number)
    print('frame number:', frame_number)
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp_grayframe, des_grayframe = sift.detectAndCompute(grayframe,None)
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
        points = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1, 1, 2)
        distort_pts = cv2.perspectiveTransform(points, matrix)
        homography = cv2.polylines(frame, [np.int32(distort_pts)], True, (255, 0, 0), 3)
        print('Width(x) in pixel = ',np.int32(distort_pts)[3][0][0]-np.int32(distort_pts)[0][0][0])
        x_pixel = np.int32(distort_pts)[3][0][0]-np.int32(distort_pts)[0][0][0]
        # distance(x_pixel)
        if frame_number == 0:
            x = x_pixel
            print('x',x)
            print('-----------------------------')



        if frame_number > 0:

            x_prime = x_pixel
            print('x_prime',x_prime)
            print('x',x)
            a = x_prime / x
            tau = a / (a-1)
            print('a',a)
            print('tau',tau)
            tau_list.append(tau)
            print('-----------------------------')
            x = x_prime

        cv2.imshow('Homography', homography)


    cv2.imshow('matches_result', matches_result)

    frame_number = frame_number + 3
    # skip frame >> frame_number += 10 or 20
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    key = cv2.waitKey(1)
    if key == 27:
        break

    if frame_number == 54:
        break

# plot estimate graphic
x = np.linspace(0, 51, 17)
z = np.array(tau_list)
print('xtype',x.shape)
print('ztype',z.shape)
plt.xlabel('Frame Number')
plt.ylabel('Time to impact (tau)')

# points
plt.plot(x, z, '.')

# Regression Liner Line (only in points)
# m, b = np.polyfit(x, z, 1)
# plt.plot(x, m*x + b)

# Regression Liner Line (extension out of points)
coefficients = np.polyfit(x, z, 1)
polynomial = np.poly1d(coefficients)
x_axis = np.linspace(0,90,100)
y_axis = polynomial(x_axis)
plt.plot(x_axis, y_axis)


plt.grid()

plt.show()
cv2.destroyAllWindows()



# load txt file (code example from A2 - Task4)
# data_points = open('Data Points.txt')
#
# for line in data_points:    # deal with data line by line
#     data = [float(x) for x in line.split()]     # separate every factor by detecting block space
#     if len(data) == 2:          # two factors for image points
#         imgpoints.append(data)
#     elif len(data) == 3:        # three factors for object points
#         objpoints.append(data)
#     else:
#         exit()





