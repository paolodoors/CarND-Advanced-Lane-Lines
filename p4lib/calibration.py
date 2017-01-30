import os
import sys
import cv2
import pickle
import numpy as np

class Calibration():
    
    def __init__(self, calibration_dir, calibration_file):
        self.calibrated = False

        if os.path.exists(calibration_file):
            # Load calibration data
            self.calibrated = True
            [self.ret, self.mtx, self.dist, self.rvecs, self.tvecs] = pickle.load(open(calibration_file, 'rb'))
            return

        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        img_size = None
        for calibration_img in os.listdir(calibration_dir):
            image_file = os.path.join(calibration_dir, calibration_img)
            img = cv2.imread(image_file)

            if not img_size:
                img_size = img.shape[0:2]

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret = False
            idx = 0
            coord = [(9, 6), (8, 6), (9, 5), (8, 5)]
            while not ret and idx < len(coord):
                nm = coord[idx]
                idx += 1
                ret, corners = cv2.findChessboardCorners(gray, nm, None)

            if ret:
                # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
                objp = np.zeros((nm[1] * nm[0], 3), np.float32)
                objp[:,:2] = np.mgrid[0:nm[0], 0:nm[1]].T.reshape(-1,2)
                # Gather points
                objpoints.append(objp)
                imgpoints.append(corners)
            else:
                print('Image {} discarded'.format(image_file))

        # Do camera calibration given object points and image points
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        self.calibrated = True
        pickle.dump([self.ret, self.mtx, self.dist, self.rvecs, self.tvecs], open(calibration_file, 'wb'))


    def undistort(self, image):
        if self.calibrated:
            return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)