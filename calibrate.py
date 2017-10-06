import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle



test_paths = glob.glob("camera_cal/test*")
test_images = []
for path in test_paths:
    test_images.append(mpimg.imread(path))



def Calibrate(show_example=False):
    paths = glob.glob("camera_cal/calibration*")

    chess_imgs = []
    for path in paths:
        chess_imgs.append(mpimg.imread(path))

    objpoints = []
    imgpoints = []

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    nx = 9
    ny = 6

    for img in chess_imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # find corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # if found, draw corners
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    if not ret:
        raise Exception("calibration unsuccessfull")
    if(show_example):
        plt.imshow(img)
        plt.figure()
    return ret, mtx, dist, rvecs, tvecs


ret, mtx, dist, rvecs, tvecs = Calibrate(show_example=True)

dst = cv2.undistort(test_images[1],mtx,dist,None,mtx)

with open('calibration.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([ret, mtx, dist, rvecs, tvecs], f)

plt.imshow(test_images[1])
plt.figure()
plt.imshow(dst)
plt.show()