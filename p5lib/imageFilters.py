#!/usr/bin/python
"""
imageFilters.py: version 0.1.0

History:
2017/01/29: coding style phase1:
            reformat to python-guide.org code style
            http://docs.python-guide.org/en/latest/writing/style/
            which uses PEP 8 as a base: http://pep8.org/.
2017/01/21: Remove line based image buffers and add road projected image
2017/01/06: Initial version converted to a class
"""

import numpy as np
import cv2
import math
from p5lib.cameraCal import CameraCal


class ImageFilters():
    # Initialize ImageFilter

    def __init__(self, camCal, projectedX, projectedY,
                 defaultThrowDistance=100.0, debug=False):
        # set debugging
        self.debug = debug

        # frameNumber
        self.curFrame = None

        # our own copy of the camera calibration results
        self.mtx, self.dist, self.img_size = camCal.get()

        # normal image size
        self.x, self.y = self.img_size

        # projected image size
        self.projectedX = projectedX
        self.projectedY = projectedY

        # mid point in picture (by height)
        self.mid = int(self.y / 2)

        # current Image RGB - undistorted
        self.curImage = np.zeros((self.y, self.x, 3), dtype=np.float32)

        # current Image Top half RGB
        self.curSkyRGB = np.zeros((self.mid, self.x, 3), dtype=np.float32)

        # current Image Bottom half RGB
        self.curRoadRGB = np.zeros((self.mid, self.x, 3), dtype=np.float32)

        # current Sky Luma Image
        self.curSkyL = np.zeros((self.mid, self.x), dtype=np.float32)

        # current Road Luma Image
        self.curRoadL = np.zeros((self.mid, self.x), dtype=np.float32)

        # current Edge
        self.curRoadEdge = np.zeros((self.mid, self.x), dtype=np.uint8)
        self.curRoadEdgeProjected = np.zeros(
            (self.projectedY, self.projectedX, 3), dtype=np.uint8)

        # current Projected image
        self.curRoadProjected = np.zeros(
            (self.projectedY, self.projectedX, 3), dtype=np.uint8)

        # image stats
        self.skylrgb = np.zeros((4), dtype=np.float32)
        self.roadlrgb = np.zeros((4), dtype=np.float32)
        self.roadbalance = 0.0
        self.horizonFound = False
        self.roadhorizon = 0
        self.visibility = 0
        self.defaultThrowDistance = defaultThrowDistance
        self.throwDistanceFound = False
        self.throwDistancePixel = self.projectedY
        self.throwDistance = defaultThrowDistance

        # Textural Image Info
        self.skyText = 'NOIMAGE'
        self.skyImageQ = 'NOIMAGE'
        self.roadText = 'NOIMAGE'
        self.roadImageQ = 'NOIMAGE'

        # set up debugging diag screens
        if self.debug:
            self.diag1 = np.zeros((self.mid, self.x, 3), dtype=np.float32)
            self.diag2 = np.zeros((self.mid, self.x, 3), dtype=np.float32)
            self.diag3 = np.zeros((self.mid, self.x, 3), dtype=np.float32)
            self.diag4 = np.zeros((self.mid, self.x, 3), dtype=np.float32)

    # Define a function to chop a picture in half horizontally
    def makehalf(self, image, half=0):
        if half == 0:
            if len(image.shape) < 3:
                newimage = np.copy(image[self.mid:self.y, :])
            else:
                newimage = np.copy(image[self.mid:self.y, :, :])
        else:
            if len(image.shape) < 3:
                newimage = np.copy(image[0:self.mid, :])
            else:
                newimage = np.copy(image[0:self.mid, :, :])
        return newimage

    # Define a function to make a half picture whole horizontally
    def makefull(self, image, half=0):
        if len(image.shape) < 3:
            newimage = np.zeros((self.y, self.x), dtype=np.uint8)
        else:
            newimage = np.zeros((self.y, self.x, 3), dtype=np.uint8)

        if half == 0:
            if len(image.shape) < 3:
                newimage[self.mid:self.y, :] = image
            else:
                newimage[self.mid:self.y, :, :] = image
        else:
            if len(image.shape) < 3:
                newimage[0:self.mid, :] = image
            else:
                newimage[0:self.mid, :, :] = image
        return newimage

    # Define a function that attempts to masks out yellow lane lines
    def image_only_yellow_white(self, image):
        # setup inRange to mask off everything except white and yellow
        lower_yellow_white = np.array([140, 140, 64])
        upper_yellow_white = np.array([255, 255, 255])
        mask = cv2.inRange(image, lower_yellow_white, upper_yellow_white)
        return cv2.bitwise_and(image, image, mask=mask)

    # Define a function that applies Gaussian Noise kernel
    def gaussian_blur(self, img, kernel_size):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # Define a function that applies Canny transform
    def canny(self, img, low_threshold, high_threshold, kernel_size):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur_gray = self.gaussian_blur(gray, kernel_size)
        return cv2.Canny(
            blur_gray.astype(np.uint8), low_threshold, high_threshold)

    # Define a function that applies Sobel x or y,
    # then takes an absolute value and applies a threshold.
    def abs_sobel_thresh(self, img, orient='x', thresh=(0, 255)):
        # Apply the following steps to img
        # 1) Convert to grayscale
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        # 3) Take the absolute value of the derivative or gradient
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        # 5) Create a mask of 1's where the scaled gradient magnitude
        #    is > thresh_min and < thresh_max
        # 6) Return this mask as your binary_output image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            abs_sobel = np.absolute(sobelx)
        if orient == 'y':
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            abs_sobel = np.absolute(sobely)
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a copy and apply the threshold
        ret, binary_output = cv2.threshold(
            scaled_sobel, thresh[0], thresh[1], cv2.THRESH_BINARY)
        # Return the result
        return binary_output

    # Define a function that applies Sobel x and y,
    # then computes the magnitude of the gradient
    # and applies a threshold
    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Calculate the magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # 5) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # 6) Create a binary mask where mag thresholds are met
        ret, mag_binary = cv2.threshold(
            gradmag, mag_thresh[0], mag_thresh[1], cv2.THRESH_BINARY)
        # 7) Return this mask as your binary_output image
        return mag_binary

    # Define a function that applies Sobel x and y,
    # then computes the direction of the gradient
    # and applies a threshold.
    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Calculate the direction of the gradient
        # 4) Take the absolute value
        with np.errstate(divide='ignore', invalid='ignore'):
            dirout = np.absolute(np.arctan(sobely / sobelx))
            # 5) Create a binary mask where direction thresholds are met
            dir_binary = np.zeros_like(dirout).astype(np.float32)
            dir_binary[(dirout > thresh[0]) & (dirout < thresh[1])] = 1
            # 6) Return this mask as your binary_output image
        # update nan to number
        np.nan_to_num(dir_binary)
        # make it fit
        dir_binary[(dir_binary > 0) | (dir_binary < 0)] = 128
        return dir_binary.astype(np.uint8)

    # Python 3 has support for cool math symbols.
    def miximg(self, img1, img2, α=0.8, β=1., λ=0.):
        """
        The result image is computed as follows:
        img1 * α + img2 * β + λ
        NOTE: img1 and img2 must be the same shape!
        """
        return cv2.addWeighted(img1.astype(np.uint8),
                               α, img2.astype(np.uint8), β, λ)

    # Define a function that thresholds the S-channel of HLS
    def hls_s(self, img, thresh=(0, 255)):
        # 1) Convert to HLS color space
        # 2) Apply a threshold to the S channel
        # 3) Return a binary image of threshold result
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s = hls[:, :, 2]
        retval, s_binary = cv2.threshold(s.astype('uint8'), thresh[
                                         0], thresh[1], cv2.THRESH_BINARY)
        return s_binary

    # Define a function that thresholds the H-channel of HLS
    def hls_h(self, img, thresh=(0, 255)):
        # 1) Convert to HLS color space
        # 2) Apply a threshold to the S channel
        # 3) Return a binary image of threshold result
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h = hls[:, :, 0]
        retval, h_binary = cv2.threshold(h.astype('uint8'), thresh[
                                         0], thresh[1], cv2.THRESH_BINARY)
        return h_binary

    # retrieve edges detected by the filter combinations (used by other
    # modules to locate current binary image.)
    def edges(self):
        # piece together images that we want to project
        img = np.zeros((self.y, self.x, 3), dtype=np.uint8)
        img[self.mid:self.y, :, :] = np.dstack(
            (self.curRoadEdge, self.curRoadEdge, self.curRoadEdge))
        return img

    # check image quality
    def imageQ(self, image):
        self.curImage = cv2.undistort(
            image, self.mtx, self.dist, None, self.mtx).astype(np.float32)
        self.yuv = cv2.cvtColor(
            self.curImage, cv2.COLOR_RGB2YUV).astype(np.float32)

        # get some stats for the sky image
        self.curSkyL = self.yuv[0:self.mid, :, 0]
        self.curSkyRGB[:, :] = self.curImage[0:self.mid, :]
        self.skylrgb[0] = np.average(self.curSkyL[0:self.mid, :])
        self.skylrgb[1] = np.average(self.curSkyRGB[0:self.mid, :, 0])
        self.skylrgb[2] = np.average(self.curSkyRGB[0:self.mid, :, 1])
        self.skylrgb[3] = np.average(self.curSkyRGB[0:self.mid, :, 2])

        # get some stats for the road image
        self.curRoadL = self.yuv[self.mid:self.y, :, 0]
        self.curRoadRGB[:, :] = self.curImage[self.mid:self.y, :]
        self.roadlrgb[0] = np.average(self.curRoadL[0:self.mid, :])
        self.roadlrgb[1] = np.average(self.curRoadRGB[0:self.mid, :, 0])
        self.roadlrgb[2] = np.average(self.curRoadRGB[0:self.mid, :, 1])
        self.roadlrgb[3] = np.average(self.curRoadRGB[0:self.mid, :, 2])

        # Sky image condition
        if self.skylrgb[0] > 160:
            self.skyImageQ = 'Sky Image: overexposed'
        elif self.skylrgb[0] < 50:
            self.skyImageQ = 'Sky Image: underexposed'
        elif self.skylrgb[0] > 143:
            self.skyImageQ = 'Sky Image: normal bright'
        elif self.skylrgb[0] < 113:
            self.skyImageQ = 'Sky Image: normal dark'
        else:
            self.skyImageQ = 'Sky Image: normal'

        # Sky detected weather or lighting conditions
        if self.skylrgb[0] > 128:
            if self.skylrgb[3] > self.skylrgb[0]:
                if self.skylrgb[1] > 120 and self.skylrgb[2] > 120:
                    if (self.skylrgb[2] - self.skylrgb[1]) > 20.0:
                        self.skyText = 'Sky Condition: tree shaded'
                    else:
                        self.skyText = 'Sky Condition: cloudy'
                else:
                    self.skyText = 'Sky Condition: clear'
            else:
                self.skyText = 'Sky Condition: UNKNOWN SKYL>128'
        else:
            if self.skylrgb[2] > self.skylrgb[3]:
                self.skyText = 'Sky Condition: surrounded by trees'
                self.visibility = -80
            elif self.skylrgb[3] > self.skylrgb[0]:
                if (self.skylrgb[2] - self.skylrgb[1]) > 10.0:
                    self.skyText = 'Sky Condition: tree shaded'
                else:
                    self.skyText = \
                        'Sky Condition: very cloudy or under overpass'
            else:
                self.skyText = 'Sky Condition: UNKNOWN!'

        self.roadbalance = self.roadlrgb[0] / 10.0

        # Road image condition
        if self.roadlrgb[0] > 160:
            self.roadImageQ = 'Road Image: overexposed'
        elif self.roadlrgb[0] < 50:
            self.roadImageQ = 'Road Image: underexposed'
        elif self.roadlrgb[0] > 143:
            self.roadImageQ = 'Road Image: normal bright'
        elif self.roadlrgb[0] < 113:
            self.roadImageQ = 'Road Image: normal dark'
        else:
            self.roadImageQ = 'Road Image: normal'

    # function to detect the horizon using the Sobel magnitude operation
    def horizonDetect(self, debug=False, thresh=50):
        if not self.horizonFound:
            img = np.copy(self.curRoadRGB).astype(np.uint8)
            magch = self.mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 150))
            horizonLine = 50
            while not self.horizonFound and horizonLine < int(self.y / 2):
                magchlinesum = np.sum(
                    magch[horizonLine:(horizonLine + 1), :]).astype(np.float32)
                if magchlinesum > (self.x * thresh):
                    self.horizonFound = True
                    self.roadhorizon = horizonLine + int(self.y / 2)
                    if debug:
                        self.diag4[horizonLine:(horizonLine + 1), :, 0] = 255
                        self.diag4[horizonLine:(horizonLine + 1), :, 1] = 255
                        self.diag4[horizonLine:(horizonLine + 1), :, 2] = 0
                else:
                    horizonLine += 1

    # function to detect the throw distance of the projection
    def projectionThrowDistanceDetect(self, debug=False, thresh=150.0):
        # maxThrowSum = 0
        if not self.throwDistanceFound:
            maskedEdge = np.copy(self.curRoadEdgeProjected[
                                 :, :, 1]).astype(np.uint8)
            topOfThrow = 0
            while not self.throwDistanceFound and \
                    topOfThrow < int(self.projectedY * 0.75):
                maskedEdgeLineSum = np.sum(
                     maskedEdge[topOfThrow:(topOfThrow + 1), :])
                maskedEdgeLineSum = maskedEdgeLineSum.astype(np.float32)

                # if maxThrowSum < maskedEdgeLineSum:
                #    maxThrowSum = maskedEdgeLineSum
                if maskedEdgeLineSum > thresh:
                    self.throwDistanceFound = True
                    self.throwDistancePixel = topOfThrow
                    self.throwDistance = self.throwDistance * \
                        ((self.projectedY - topOfThrow) / self.projectedY)
                    if debug:
                        self.curRoadEdgeProjected[
                            topOfThrow:(topOfThrow + 1), :, 0] = 0
                        self.curRoadEdgeProjected[
                            topOfThrow:(topOfThrow + 1), :, 1] = 255
                        self.curRoadEdgeProjected[
                            topOfThrow:(topOfThrow + 1), :, 2] = 0
                else:
                    topOfThrow += 1
        # print("maxThrowSum: ", maxThrowSum)
        return self.throwDistancePixel

    # function to attempt to balance the image exposure for easier lane line
    # detection
    def balanceEx(self):
        # separate each of the RGB color channels
        r = self.curRoadRGB[:, :, 0]
        g = self.curRoadRGB[:, :, 1]
        b = self.curRoadRGB[:, :, 2]
        # Get the Y channel (Luma) from the YUV color space
        # and make two copies
        yo = np.copy(self.curRoadL[:, :]).astype(np.float32)
        yc = np.copy(self.curRoadL[:, :]).astype(np.float32)
        # use the balance factor calculated previously to calculate the
        # corrected Y
        yc = (yc / self.roadbalance) * 8.0
        # make a copy and threshold it to maximum value 255.
        lymask = np.copy(yc)
        lymask[(lymask > 255.0)] = 255.0
        # create another mask that attempts to masks yellow road markings.
        uymask = np.copy(yc) * 0
        # subtract the thresholded mask from the corrected Y.
        # Now we just have peaks.
        yc -= lymask
        # If we are dealing with an over exposed image
        # cap its corrected Y to 242.
        if self.roadlrgb[0] > 160:
            yc[(b > 254) & (g > 254) & (r > 254)] = 242.0
        # If we are dealing with a darker image
        # try to pickup faint blue and cap them to 242.
        elif self.roadlrgb[0] < 128:
            yc[(b > self.roadlrgb[3]) & (
                yo > 160 + (self.roadbalance * 20))] = 242.0
        else:
            yc[(b > self.roadlrgb[3]) & (
                yo > 210 + (self.roadbalance * 10))] = 242.0
        # attempt to mask yellow lane lines
        uymask[(b < self.roadlrgb[0]) & (r > self.roadlrgb[0]) &
               (g > self.roadlrgb[0])] = 242.0
        # combined the corrected road luma and the masked yellow
        yc = self.miximg(yc, uymask, 1.0, 1.0)
        # mix it back to the original luma.
        yc = self.miximg(yc, yo, 1.0, 0.8)
        # resize the image in an attempt to get the lane lines to the bottom.
        yc[int((self.y / 72) * 70):self.y, :] = 0
        self.yuv[self.mid:self.y, :, 0] = yc.astype(np.uint8)
        self.yuv[(self.y - 40):self.y, :, 0] = \
            yo[(self.mid - 40):self.mid, :].astype(np.uint8)
        # convert back to RGB.
        self.curRoadRGB = cv2.cvtColor(
            self.yuv[self.mid:self.y, :, :], cv2.COLOR_YUV2RGB)

    # filter 1
    # builds combination number 1
    def applyFilter1(self):
        # Run the functions
        img = np.copy(self.curRoadRGB).astype(np.uint8)
        gradx = self.abs_sobel_thresh(img, orient='x', thresh=(25, 100))
        grady = self.abs_sobel_thresh(img, orient='y', thresh=(50, 150))
        magch = self.mag_thresh(img, sobel_kernel=9, mag_thresh=(50, 250))
        dirch = self.dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
        sch = self.hls_s(img, thresh=(88, 190))
        hch = self.hls_h(img, thresh=(50, 100))

        # Output "masked_lines" is a single channel mask
        shadow = np.zeros_like(dirch).astype(np.uint8)
        shadow[(sch > 0) & (hch > 0)] = 128

        # create the Red filter
        rEdgeDetect = img[:, :, 0] / 4
        rEdgeDetect = 255 - rEdgeDetect
        rEdgeDetect[(rEdgeDetect > 210)] = 0

        # build the combination
        combined = np.zeros_like(dirch).astype(np.uint8)
        combined[((gradx > 0) | (grady > 0) | ((magch > 0) & (dirch > 0)) | (
            sch > 0)) & (shadow == 0) & (rEdgeDetect > 0)] = 35
        self.curRoadEdge = combined

        # build diag screen if in debug mode
        if self.debug:
            # create diagnostic screen 1-3
            # creating a blank color channel for combining
            ignore_color = np.copy(gradx) * 0
            self.diag1 = np.dstack((rEdgeDetect, gradx, grady))
            self.diag2 = np.dstack((ignore_color, magch, dirch))
            self.diag3 = np.dstack((sch, shadow, hch))
            self.diag4 = np.dstack((combined, combined, combined)) * 4

    # filter 2
    # builds combination number 2
    def applyFilter2(self):
        # Run the functions
        img = np.copy(self.curRoadRGB).astype(np.uint8)
        gradx = self.abs_sobel_thresh(img, orient='x', thresh=(25, 100))
        grady = self.abs_sobel_thresh(img, orient='y', thresh=(50, 150))
        magch = self.mag_thresh(img, sobel_kernel=9, mag_thresh=(50, 250))
        dirch = self.dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
        sch = self.hls_s(img, thresh=(88, 250))
        hch = self.hls_h(img, thresh=(50, 100))

        # Output "masked_lines" is a single channel mask
        shadow = np.zeros_like(dirch).astype(np.uint8)
        shadow[(sch > 0) & (hch > 0)] = 128

        # create the Red filter
        rEdgeDetect = img[:, :, 0] / 4
        rEdgeDetect = 255 - rEdgeDetect
        rEdgeDetect[(rEdgeDetect > 210)] = 0

        # build the combination
        combined = np.zeros_like(dirch).astype(np.uint8)
        combined[((gradx > 0) | (grady > 0) | ((magch > 0) & (dirch > 0)) | (
            sch > 0)) & (shadow == 0) & (rEdgeDetect > 0)] = 35
        combined[(grady > 0) & (dirch > 0) & (magch > 0)] = 35
        self.curRoadEdge = combined

        # build diag screen if in debug mode
        if self.debug:
            # create diagnostic screen 1-3
            # creating a blank color channel for combining
            ignore_color = np.copy(gradx) * 0
            self.diag1 = np.dstack((rEdgeDetect, gradx, grady))
            self.diag2 = np.dstack((ignore_color, magch, dirch))
            self.diag3 = np.dstack((sch, shadow, hch))
            self.diag4 = np.dstack((combined, combined, combined)) * 4

    # filter 3
    # builds combination number 3
    def applyFilter3(self):
        # Run the functions
        img = np.copy(self.curRoadRGB).astype(np.uint8)
        gradx = self.abs_sobel_thresh(img, orient='x', thresh=(25, 100))
        grady = self.abs_sobel_thresh(img, orient='y', thresh=(50, 150))
        magch = self.mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 150))
        dirch = self.dir_threshold(img, sobel_kernel=15, thresh=(0.6, 1.3))
        sch = self.hls_s(img, thresh=(20, 100))
        hch = self.hls_h(img, thresh=(125, 175))

        # create the Red filter
        rEdgeDetect = img[:, :, 0] / 4
        rEdgeDetect = 255 - rEdgeDetect
        rEdgeDetect[(rEdgeDetect > 220)] = 0

        # Output "masked_lines" is a single channel mask
        shadow = np.zeros_like(dirch).astype(np.uint8)
        shadow[(sch > 0) & (hch > 0)] = 128

        # build the combination
        combined = np.zeros_like(dirch).astype(np.uint8)
        combined[((gradx > 0) | (grady > 0) | ((magch > 0) & (dirch > 0)) | (
            sch > 0)) & (shadow == 0) & (rEdgeDetect > 0)] = 35
        self.curRoadEdge = combined

        # build diag screen if in debug mode
        if self.debug:
            # create diagnostic screen 1-3
            # creating a blank color channel for combining
            ignore_color = np.copy(gradx) * 0
            self.diag1 = np.dstack((rEdgeDetect, gradx, grady))
            self.diag2 = np.dstack((ignore_color, magch, dirch))
            self.diag3 = np.dstack((sch, shadow, hch))
            self.diag4 = np.dstack((combined, combined, combined)) * 4

    # filter 4
    # builds combination number 4
    def applyFilter4(self):
        # Run the functions
        img = np.copy(self.curRoadRGB).astype(np.uint8)
        gradx = self.abs_sobel_thresh(img, orient='x', thresh=(30, 100))
        grady = self.abs_sobel_thresh(img, orient='y', thresh=(75, 150))
        magch = self.mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 150))
        dirch = self.dir_threshold(img, sobel_kernel=15, thresh=(0.6, 1.3))
        sch = self.hls_s(img, thresh=(20, 100))
        hch = self.hls_h(img, thresh=(125, 175))

        # create the Red filter
        rEdgeDetect = img[:, :, 0] / 4
        rEdgeDetect = 255 - rEdgeDetect
        rEdgeDetect[(rEdgeDetect > 220)] = 0

        # Output "masked_lines" is a single channel mask
        shadow = np.zeros_like(dirch).astype(np.uint8)
        shadow[(sch > 0) & (hch > 0)] = 128

        # build the combination
        combined = np.zeros_like(dirch).astype(np.uint8)
        combined[((magch > 0) & (dirch > 0)) | (
            (rEdgeDetect > 192) & (rEdgeDetect < 200) & (magch > 0))] = 35
        self.curRoadEdge = combined

        # build diag screen if in debug mode
        if self.debug:
            # create diagnostic screen 1-3
            # creating a blank color channel for combining
            ignore_color = np.copy(gradx) * 0
            self.diag1 = np.dstack((rEdgeDetect, gradx, grady))
            self.diag2 = np.dstack((ignore_color, magch, dirch))
            self.diag3 = np.dstack((sch, shadow, hch))
            self.diag4 = np.dstack((combined, combined, combined)) * 4

    # filter 5
    # builds combination number 5
    def applyFilter5(self):
        # Run the functions
        img = np.copy(self.curRoadRGB).astype(np.uint8)
        gradx = self.abs_sobel_thresh(img, orient='x', thresh=(25, 100))
        grady = self.abs_sobel_thresh(img, orient='y', thresh=(50, 150))
        magch = self.mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 150))
        dirch = self.dir_threshold(img, sobel_kernel=15, thresh=(0.5, 1.3))
        sch = self.hls_s(img, thresh=(20, 80))
        hch = self.hls_h(img, thresh=(130, 175))

        # create the Red filter
        rEdgeDetect = img[:, :, 0] / 4
        rEdgeDetect = 255 - rEdgeDetect
        rEdgeDetect[(rEdgeDetect > 220)] = 0

        # Output "masked_lines" is a single channel mask
        shadow = np.zeros_like(dirch).astype(np.uint8)
        shadow[(sch > 0) & (hch > 0)] = 128

        # build the combination
        combined = np.zeros_like(dirch).astype(np.uint8)
        combined[(rEdgeDetect > 192) & (rEdgeDetect < 205) & (sch > 0)] = 35
        self.curRoadEdge = combined

        # build diag screen if in debug mode
        if self.debug:
            # create diagnostic screen 1-3
            # creating a blank color channel for combining
            ignore_color = np.copy(gradx) * 0
            self.diag1 = np.dstack((rEdgeDetect, gradx, grady))
            self.diag2 = np.dstack((ignore_color, magch, dirch))
            self.diag3 = np.dstack((sch, shadow, hch))
            self.diag4 = np.dstack((combined, combined, combined)) * 4

    # set the edge projection image
    def setEdgeProjection(self, projected):
        # print("projected: ",projected.shape)
        self.curRoadEdgeProjected = np.copy(projected)

    # get the edge projection image
    def getEdgeProjection(self):
        return self.curRoadEdgeProjected

    # set the full road projection image
    def setRoadProjection(self, projected):
        self.curRoadProjected = np.copy(projected)

    # get the full road projection image
    def getRoadProjection(self):
        return self.curRoadProjected

    # draw the discovered horizon in the image
    def drawHorizon(self, image):
        horizonLine = self.roadhorizon
        image[horizonLine:(horizonLine + 1), :, 0] = 255
        image[horizonLine:(horizonLine + 1), :, 1] = 255
        image[horizonLine:(horizonLine + 1), :, 2] = 0
