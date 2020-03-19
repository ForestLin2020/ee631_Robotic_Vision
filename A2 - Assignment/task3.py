import numpy as np
import cv2

mtx = np.load('intrinsic  parameters.npy')
dist = np.load('distortion parameters.npy')

# Function of processing three pictures
def different(name):
    # Get picture
    img = cv2.imread(name + '.jpg')

    # Refine the camera matrix
    h, w = img.shape[:2]

    # windows python 3.7 need to +1, because img.shape change the size of image
    # ask why this happen on windows python but not on linux python
    h += 1
    w += 1

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite(name+'.png', dst)

    # Compare Original one and Distortion one
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    cv2.imwrite(name + 'diff' + '.jpg', diff)

    cv2.imshow(name + ' Differet Detection', diff)
    return

different('Close')
different('Far')
different('Turn')

cv2.waitKey(0)
cv2.destroyAllWindows()