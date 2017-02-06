#!/usr/bin/python
"""
cameraCal.py: version 0.1.0

History:
2017/01/29: coding style phase1:
            reformat to python-guide.org code style
            http://docs.python-guide.org/en/latest/writing/style/
            which uses PEP 8 as a base: http://pep8.org/.
2017/01/06: Initial version converted to a class
"""

import numpy as np
import cv2
import glob
import re
import os
import pickle


class CameraCal():

    # initialize - either go through and calculate the camera
    # calibration if no pickle file exists
    # or just load the pickle file.
    def __init__(self, calibration_dir, pickle_file):
        # Initialize cameraCal
        self.mtx = None
        self.dist = None
        self.img_size = None

        if not os.path.isfile(pickle_file):
            objpoints = []  # 3d points in real world space
            imgpoints = []  # 2d points in image plane.

            # prepare object points: (0,0,0), (1,0,0), (2,0,0) .., (6,5,0)
            # The images may have different detected checker board dimensions!
            # Currently, possible dimension combinations are: (9,6), (8,6),
            # (9,5), (9,4) and (7,6)
            objp1 = np.zeros((6 * 9, 3), np.float32)
            objp1[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
            objp2 = np.zeros((6 * 8, 3), np.float32)
            objp2[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
            objp3 = np.zeros((5 * 9, 3), np.float32)
            objp3[:, :2] = np.mgrid[0:9, 0:5].T.reshape(-1, 2)
            objp4 = np.zeros((4 * 9, 3), np.float32)
            objp4[:, :2] = np.mgrid[0:9, 0:4].T.reshape(-1, 2)
            objp5 = np.zeros((6 * 7, 3), np.float32)
            objp5[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
            objp6 = np.zeros((6 * 5, 3), np.float32)
            objp6[:, :2] = np.mgrid[0:5, 0:6].T.reshape(-1, 2)

            text = 'Performing camara calibrations against chessboard images: '
            print('{}"./{}/calibration*.jpg"...'.format(text, calibration_dir))
            # Make a list of calibration images
            images = glob.glob(calibration_dir + '/calibration*.jpg')

            # Step through the list and search for chessboard corners
            for idx, fname in enumerate(images):
                img = cv2.imread(fname)
                img2 = np.copy(img)
                self.img_size = (img.shape[1], img.shape[0])
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Find the chessboard corners using possible combinations of
                # dimensions.
                ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
                objp = objp1
                if not ret:
                    ret, corners = cv2.findChessboardCorners(
                        gray, (8, 6), None)
                    objp = objp2
                if not ret:
                    ret, corners = cv2.findChessboardCorners(
                        gray, (9, 5), None)
                    objp = objp3
                if not ret:
                    ret, corners = cv2.findChessboardCorners(
                        gray, (9, 4), None)
                    objp = objp4
                if not ret:
                    ret, corners = cv2.findChessboardCorners(
                        gray, (7, 6), None)
                    objp = objp5
                if not ret:
                    ret, corners = cv2.findChessboardCorners(
                        gray, (5, 6), None)
                    objp = objp6
                # print("corners: ", corners.shape, "\n", corners)

                # If found, add object points, image points
                if ret:
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    cv2.drawChessboardCorners(img2,
                                              (corners.shape[1],
                                               corners.shape[0]),
                                              corners, ret)
                    ret, self.mtx, self.dist, self.rvecs, self.tvecs = \
                        cv2.calibrateCamera(objpoints, imgpoints,
                                            self.img_size, None, None)

            # done and found all chessboard corners.
            # now time to save the results into a pickle file for later
            # retrieval without additional calculations.
            try:
                with open(pickle_file, 'w+b') as pfile1:
                    text = 'Saving data to pickle file'
                    print('{}: {} ...'.format(text, pickle_file))
                    pickle.dump({'img_size': self.img_size,
                                 'mtx': self.mtx,
                                 'dist': self.dist,
                                 'rvecs': self.rvecs,
                                 'tvecs': self.tvecs},
                                pfile1, pickle.HIGHEST_PROTOCOL)
                    print("Camera Calibration Data saved to", pickle_file)
            except Exception as e:
                print('Unable to save data to', pickle_file, ':', e)
                raise

        # previously saved pickle file of the distortion correction data
        # has been found.  go ahead and revive it.
        else:
            try:
                with open(pickle_file, 'rb') as f:
                    pickle_data = pickle.load(f)
                    self.img_size = pickle_data['img_size']
                    self.mtx = pickle_data['mtx']
                    self.dist = pickle_data['dist']
                    self.rvecs = pickle_data['rvecs']
                    self.tvecs = pickle_data['tvecs']
                    del pickle_data
                    print("Camera Calibration data restored from", pickle_file)
            except Exception as e:
                print('Unable to restore camera calibration data from',
                      pickle_file, ':', e)
                raise

    # if the source image is now smaller than the original calibration image
    # just set it
    def setImageSize(self, img_shape):
        self.img_size = (img_shape[1], img_shape[0])

    # Get a subset of the camera calibration result that
    # the rest of the pipeline wants
    def get(self):
        return self.mtx, self.dist, self.img_size

    # Get all of the camera calibration result that
    # the rest of the pipeline wants
    def getall(self):
        return self.mtx, self.dist, self.img_size, self.rvecs, self.tvecs
