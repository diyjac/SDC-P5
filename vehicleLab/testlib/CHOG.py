#!/usr/bin/python
"""
CHOG.py: version 0.1.0

History:
2017/01/23: Initial version converted to a class
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.externals import joblib
from testlib.slidingWindows import SlidingWindows

class CHOG():

    # initialize
    def __init__(self, trained_scalar=None, orient=9, pix_per_cell=8, cell_per_block=2):
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        if trained_scalar is not None:
            self.X_scaler = joblib.load(trained_scalar)

    # Define a function to compute binned color features
    def bin_spatial(self, img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features

    # Define a function to compute color histogram features
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    # Define a function to return HOG features and visualization
    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block,
                            vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            return features

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self, image, cspace='RGB', spatial_size=(32, 32),
                            hist_bins=32, hist_range=(0, 256), orient=9,
                            pix_per_cell=8, cell_per_block=2, hog_channel=0):

        if image.shape[0]>0 and image.shape[1]>0:
            if image.shape[0]!=64 or image.shape[1]!=64:
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
            else: feature_image = np.copy(image)
            # Apply bin_spatial() to get spatial color features
            if cspace == 'GRAYRGB':
                spatial_features = self.bin_spatial(rgbfeature_image, size=spatial_size)
                # Apply color_hist() also with a color space option now
                hist_features = self.color_hist(rgbfeature_image, nbins=hist_bins, bins_range=hist_range)
                # Call get_hog_features() with vis=False, feature_vec=True
                hog_features = self.get_hog_features(feature_image, orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                # Append the new feature vector to the features list
                hogFeatures = np.concatenate((spatial_features, hist_features, hog_features))
            elif cspace == 'GRAY':
                hog_features = self.get_hog_features(feature_image, orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                hogFeatures = hog_features
            else:
                spatial_features = self.bin_spatial(feature_image, size=spatial_size)
                # Apply color_hist() also with a color space option now
                hist_features = self.color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
                # Call get_hog_features() with vis=False, feature_vec=True
                hog_features = self.get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                # Append the new feature vector to the features list
                hogFeatures = np.concatenate((spatial_features, hist_features, hog_features))
            return self.X_scaler.transform(hogFeatures.reshape(1, -1))
        else:
            return None

    # Here is your draw_boxes function from the previous exercise
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    # Define a way for us to write out a sample of the HOG
    def drawPlots(self, imagefile, sampleTitle, images, hconf, lconf):
        print("saving image and hog results to ", imagefile)
        # Setup plot
        fig = plt.figure(figsize=(12, len(images)*9))
        w_ratios = [2.0, 6.5, 6.5]
        h_ratios = [9.0 for n in range(len(images))]
        grid = gridspec.GridSpec(len(images), 3, wspace=0.05, hspace=0.0, width_ratios=w_ratios, height_ratios=h_ratios)
        i = 0

        for filename, orient, pix_per_cell, cell_per_block, image1, image2 in images:
            # draw the images
            # next image
            title = '%s\n Orientation: %d\n Pix_per_cell: %d\n Cell_per_block: %d\n Confidence Range:\n  High: %10.5f\n   Low: %10.5f'%(filename, orient, pix_per_cell, cell_per_block, hconf, lconf)

            ax = plt.Subplot(fig, grid[i])
            ax.text(-0.5,0.4, title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            fig.add_subplot(ax)
            i += 1

            ax = plt.Subplot(fig, grid[i])
            ax.imshow(image1)
            if i==1:
                ax.set_title('Augmented %s High'%(sampleTitle), size=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            i += 1

            ax = plt.Subplot(fig, grid[i])
            ax.imshow(image2)
            if i==2:
                ax.set_title('Augmented %s Low'%(sampleTitle), size=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            i += 1

        plt.savefig(imagefile)
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
        y,x,ch = image.shape
        cuttoff = int((y/len(images))*0.65)
        image = image[cuttoff:(y-cuttoff),:,:]
        cv2.imwrite(imagefile, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Define a way for us to write out a sample of the HOG
    def drawXOPlots(self, imagefile, sampleTitle, images):
        print("saving image and hog results to ", imagefile)
        # Setup plot
        fig = plt.figure(figsize=(12, len(images)*9))
        w_ratios = [2.0, 6.5, 6.5]
        h_ratios = [9.0 for n in range(len(images))]
        grid = gridspec.GridSpec(len(images), 3, wspace=0.05, hspace=0.0, width_ratios=w_ratios, height_ratios=h_ratios)
        i = 0

        for filename, orient, pix_per_cell, cell_per_block, image1, image2 in images:
            # draw the images
            # next image
            title = '%s\n Orientation: %d\n Pix_per_cell: %d\n Cell_per_block: %d'%(filename, orient, pix_per_cell, cell_per_block)

            ax = plt.Subplot(fig, grid[i])
            ax.text(-0.5,0.4, title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            fig.add_subplot(ax)
            i += 1

            ax = plt.Subplot(fig, grid[i])
            ax.imshow(image1)
            if i==1:
                ax.set_title('Original', size=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            i += 1

            ax = plt.Subplot(fig, grid[i])
            ax.imshow(image2)
            if i==2:
                ax.set_title('Augmented %s'%(sampleTitle), size=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            i += 1

        plt.savefig(imagefile)
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
        y,x,ch = image.shape
        cuttoff = int((y/len(images))*0.65)
        image = image[cuttoff:(y-cuttoff),:,:]
        cv2.imwrite(imagefile, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


