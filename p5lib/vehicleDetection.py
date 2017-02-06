#!/usr/bin/python
"""
vehicleDetection.py: version 0.1.0

History:
2017/01/29: coding style phase1:
    reformat to python-guide.org code style
    http://docs.python-guide.org/en/latest/writing/style/
    which uses PEP 8 as a base: http://pep8.org/.
2017/01/23: Initial version converted to a class
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
import glob
import time
import os
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.externals import joblib
from p5lib.roadGrid import RoadGrid


# a class for wrapping our SVM trained HOG vehicle detector.
class VehicleDetection():
    # initialize
    def __init__(self, projectedX, projectedY, versionName=None,
                 cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2,
                 hog_channel=0, threshold=2.5,
                 dataFileNamePattern="imgExt%03d.jpg"):
        self.start = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        self.projectedX = projectedX
        self.projectedY = projectedY
        self.versionName = versionName
        self.cspace = cspace
        self.hog_channel = hog_channel
        if versionName is not None:
            self.trained_model = './trained/' + versionName + '.pkl'
            self.trained_scalar = './trained/scaler' + versionName + '.pkl'
            self.svc = joblib.load(self.trained_model)
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        if self.trained_scalar is not None and \
                self.versionName is not None:
            self.X_scaler = joblib.load(self.trained_scalar)
        self.threshold = threshold
        self.dataFileNamePattern = dataFileNamePattern

    # Define a function to change the detector's threshold
    def set_threshold(self, new_threshold):
        self.threshold = new_threshold

    # Define a function to compute binned color features
    def bin_spatial(self, img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features

    # Define a function to compute color histogram features
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(
            img[:, :, 0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(
            img[:, :, 1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(
            img[:, :, 2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate(
            (channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    # Define a function to return HOG features and visualization
    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block,
                         vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis:
            features, hog_image = hog(
                img, orientations=orient,
                pixels_per_cell=(pix_per_cell, pix_per_cell),
                cells_per_block=(cell_per_block, cell_per_block),
                transform_sqrt=True,
                visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(
                img, orientations=orient,
                pixels_per_cell=(pix_per_cell, pix_per_cell),
                cells_per_block=(cell_per_block, cell_per_block),
                transform_sqrt=True, visualise=vis,
                feature_vector=feature_vec)
            return features

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self, image, cspace='RGB', spatial_size=(32, 32),
                         hist_bins=32, hist_range=(0, 256), orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel=0):

        if image.shape[0] > 0 and image.shape[1] > 0:
            if image.shape[0] != 64 or image.shape[1] != 64:
                image = cv2.resize(image, (64, 64))

            # Create a list to append feature vectors to
            # apply color conversion if other than 'RGB'
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif cspace == 'GRAY':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                elif cspace == 'GRAYRGB':
                    rgbfeature_image = np.copy(image)
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                feature_image = np.copy(image)
            # Apply bin_spatial() to get spatial color features
            if cspace == 'GRAYRGB':
                spatial_features = self.bin_spatial(
                    rgbfeature_image, size=spatial_size)
                # Apply color_hist() also with a color space option now
                hist_features = self.color_hist(
                    rgbfeature_image, nbins=hist_bins,
                    bins_range=hist_range)
                # Call get_hog_features() with vis=False, feature_vec=True
                hog_features = self.get_hog_features(
                    feature_image, orient, pix_per_cell,
                    cell_per_block, vis=False, feature_vec=True)
                # Append the new feature vector to the features list
                hogFeatures = np.concatenate(
                    (spatial_features, hist_features, hog_features))
            elif cspace == 'GRAY':
                hog_features = self.get_hog_features(
                    feature_image, orient, pix_per_cell,
                    cell_per_block, vis=False, feature_vec=True)
                hogFeatures = hog_features
            else:
                spatial_features = self.bin_spatial(
                    feature_image, size=spatial_size)
                # Apply color_hist() also with a color space option now
                hist_features = self.color_hist(
                    feature_image, nbins=hist_bins, bins_range=hist_range)
                # Call get_hog_features() with vis=False, feature_vec=True
                hog_features = self.get_hog_features(
                    feature_image[:, :, hog_channel], orient, pix_per_cell,
                    cell_per_block, vis=False, feature_vec=True)
                # Append the new feature vector to the features list
                hogFeatures = np.concatenate(
                    (spatial_features, hist_features, hog_features))
            return self.X_scaler.transform(hogFeatures.reshape(1, -1))
        else:
            return None

    # specialized sliding window generation.
    # we are looking at top down birds-eye view and
    # limiting the detection to just the lanes.
    # we need to use the lane lines to help generate the sliding window
    # locations.
    def slidingWindows(self, lines, laneIdx, complete=False):
        # calculate the window positions
        nlanes = len(lines) - 1
        x0 = self.projectedX/2
        y0 = self.projectedY

        # create roadgrid for boxes
        window_list = RoadGrid(x0, y0, nlanes, laneIdx)

        for i in range(nlanes):
            lane_boxes = {}
            leftPolynomial = np.poly1d(lines[i].currentFit)
            rightPolynomial = np.poly1d(lines[i + 1].currentFit)

            # horizontal lines
            # we treat left and right lanes differently because of the
            # projection.  In the 'complete' case we are getting all
            # of the sliding windows
            if complete:
                if i < laneIdx:
                    indexedBottom = i + 1
                else:
                    indexedBottom = i
                for j in range(
                        int(lines[indexedBottom].bottomProjectedY / 32)):
                    y1 = 32 * j
                    mid = int(
                        (rightPolynomial([y1]) +
                         leftPolynomial([y1])) / 2)
                    x1 = mid - 32
                    x2 = mid + 32
                    y2 = y1 + 64
                    if (x1 > 0 and x2 < self.projectedX and
                            y1 > 0 and y2 < self.projectedY):
                        lane_boxes['%d' % (j)] = ((x1, y1), (x2, y2))

            # In the else case we are getting only the windows at the top
            # and bottom of our lanes for the sliding windows
            else:
                linetop = lines[i].getTopPoint()
                if i == laneIdx:
                    ylist = [(linetop[1], 0),
                             (linetop[1] + 32, 1),
                             (linetop[1] + 64, 2)]
                elif i < laneIdx:
                    ylist = [(linetop[1], 0),
                             (linetop[1] + 32, 1),
                             (linetop[1] + 64, 2),
                             (lines[i].bottomProjectedY - 96, 55)]
                else:
                    ylist = [(linetop[1], 0),
                             (linetop[1] + 32, 1),
                             (linetop[1] + 64, 2),
                             (lines[i + 1].bottomProjectedY - 32, 55)]

                for y1, j in ylist:
                    mid = int(
                        (rightPolynomial([y1]) + leftPolynomial([y1])) / 2)
                    x1 = mid - 32
                    x2 = mid + 32
                    y2 = y1 + 64
                    if (x1 > 0 and x2 < self.projectedX and
                            y1 > 0 and y2 < self.projectedY):
                        lane_boxes['%d' % (j)] = ((x1, y1), (x2, y2))
            window_list.map_boxes(i, lane_boxes)
        return window_list

    # draw_boxes function
    def draw_boxes(self, img, windows, color=(255, 255, 255), thick=20):
        # Iterate through the bounding boxes in a windows list
        for bbox in windows:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(
                img, (int(bbox[0][0]), int(bbox[0][1])),
                (int(bbox[1][0]), int(bbox[1][1])), color, thick)

    # Define a way for us to write out a sample of the HOG
    def drawPlots(self, imagefile, sampleTitle, images):
        # print("saving image and hog results to ", imagefile)
        # Setup plot
        fig = plt.figure(figsize=(12, len(images) * 9))
        w_ratios = [2.0, 6.5, 6.5]
        h_ratios = [9.0 for n in range(len(images))]
        grid = gridspec.GridSpec(
            len(images), 3, wspace=0.05, hspace=0.0,
            width_ratios=w_ratios, height_ratios=h_ratios)
        i = 0

        for filename, orient, pix_per_cell, \
                cell_per_block, image1, image2 in images:
            # draw the images
            # next image
            title = '%s\n Orientation: %d\n'
            title += ' Pix_per_cell: %d\n'
            title += ' Cell_per_block: %d'
            title = title % \
                (filename, orient, pix_per_cell, cell_per_block)

            ax = plt.Subplot(fig, grid[i])
            ax.text(-0.5, 0.4, title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            fig.add_subplot(ax)
            i += 1

            ax = plt.Subplot(fig, grid[i])
            ax.imshow(image1)
            if i == 1:
                ax.set_title('Original', size=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            i += 1

            ax = plt.Subplot(fig, grid[i])
            ax.imshow(image2)
            if i == 2:
                ax.set_title('Augmented %s' % (sampleTitle), size=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            i += 1

        plt.savefig(imagefile)
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
        y, x, ch = image.shape
        cuttoff = int((y / len(images)) * 0.65)
        image = image[cuttoff:(y - cuttoff), :, :]
        cv2.imwrite(imagefile, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Define a way for us to process an image with
    # a list of sliding windows and try to detect vehicles
    def detectVehicles(self, image, roadgrid):
        mapping = roadgrid.getMapping()
        for box in mapping.keys():
            if not mapping[box]['occluded'] and \
               not mapping[box]['found'] and \
                    mapping[box]['vehicle'] is None:
                window = mapping[box]['window']
                wimage = image[
                    window[0][1]:window[1][1],
                    window[0][0]:window[1][0]]
                wfeatures = self.extract_features(
                    wimage, cspace=self.cspace, spatial_size=(32, 32),
                    orient=self.orient, pix_per_cell=self.pix_per_cell,
                    cell_per_block=self.cell_per_block,
                    hog_channel=self.hog_channel,
                    hist_bins=32, hist_range=(0, 256))
                if wfeatures is not None:
                    confidence = self.svc.decision_function(
                        wfeatures.reshape(1, -1))
                    if confidence[0] > self.threshold:
                        roadgrid.setFound(box)
        return roadgrid

    # Define a way for us to collect data from images and videos
    def collectData(self, frame, image, windows):
        baseDir = "collected/%s/%04d/" % (self.start, frame)
        if not os.path.exists(baseDir):
            os.makedirs(baseDir)
        i = 0
        for window in [lane for lane in windows]:
            wimage = image[window[0][1]:window[
                1][1], window[0][0]:window[1][0]]
            outfilename = baseDir + self.dataFileNamePattern % (i)
            cv2.imwrite(outfilename,
                        cv2.cvtColor(wimage, cv2.COLOR_RGB2BGR))
            i += 1
