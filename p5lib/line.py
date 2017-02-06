#!/usr/bin/python
"""
line.py: version 0.1.0

History:
2017/01/29: coding style phase1:
            reformat to python-guide.org code style
            http://docs.python-guide.org/en/latest/writing/style/
            which uses PEP 8 as a base: http://pep8.org/.
2017/01/07: Initial version converted to a class
"""

import numpy as np
import cv2
import math
from p5lib.lane import Lane
from p5lib.imageFilters import ImageFilters

# Define a class to receive the characteristics of each line detection


class Line():

    def __init__(self, side, x, y, projectedX, projectedY, maskDelta, n=10):
        # iterations to keep
        self.n = n

        # assigned side
        self.side = side

        # dimensions
        self.mid = int(y / 2)
        self.x = x
        self.y = y
        self.projectedX = projectedX
        self.projectedY = projectedY

        # frameNumber
        self.currentFrame = None

        # was the line detected in the last iteration?
        self.detected = False

        # was the line detected in the last iteration?
        self.confidence = 0.0
        self.confidence_based = 0

        # polynomial coefficients averaged over the last n iterations
        self.bestFit = None

        # polynomial coefficients for the most recent fit
        self.currentFit = None

        # x values of the current fitted line
        self.currentX = None

        # radius of curvature of the line in meters
        self.radiusOfCurvature = None

        # distance in meters of vehicle center from the line
        self.lineBasePos = None

        # pixel base position
        self.pixelBasePos = None
        self.bottomProjectedY = projectedY

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # x values for detected line pixels
        self.allX = None

        # y values for detected line pixels
        self.allY = None

        # xy values for drawing
        self.XYPolyline = None

        # mask delta for masking points in lane
        self.maskDelta = maskDelta

        # poly for fitting new values
        self.linePoly = None

        # mask for lanes
        self.linemask = np.zeros(
            (self.projectedY, self.projectedX), dtype=np.uint8)

        # road manager request
        self.newYTop = None

        # classify line
        self.lineClassified = False
        self.lineType = "not found"
        self.line_color = ""

    # create adjacent lane lines using an existing lane on the right
    def createPolyFitLeft(self, curImgFtr, rightLane,
                          faint=1.0, resized=False):
        # create new left line polynomial
        polyDiff = np.polysub(rightLane.lines[rightLane.left].currentFit,
                              rightLane.lines[rightLane.right].currentFit)
        self.currentFit = np.polyadd(
            rightLane.lines[rightLane.left].currentFit, polyDiff)
        polynomial = np.poly1d(self.currentFit)
        self.allY = rightLane.lines[rightLane.left].allY
        self.currentX = polynomial(self.allY)
        self.allX = self.currentX

        if len(self.allY) > 75:
            # We need to increase our pixel count by 2 to get to 100%
            # confidence and maintain the current pixel count to keep
            # the line detection
            self.confidence_based = len(self.allY) * 2
            self.confidence = len(self.allY) / self.confidence_based
            self.detected = True

            # create linepoly
            xy1 = np.column_stack(
                (self.currentX + self.maskDelta, self.allY)).astype(np.int32)
            xy2 = np.column_stack(
                (self.currentX - self.maskDelta, self.allY)).astype(np.int32)
            self.linePoly = np.concatenate((xy1, xy2[::-1]), axis=0)

            # create mask
            self.linemask = np.zeros_like(self.linemask)
            cv2.fillConvexPoly(self.linemask, self.linePoly, 64)

            # Add the point at the bottom.
            allY = np.append(self.allY, self.projectedY - 1)
            allX = polynomial(allY)
            self.XYPolyline = np.column_stack((allX, allY)).astype(np.int32)

            # create the accumulator
            self.bestFit = self.currentFit

            # classify the line
            # print("classifying the left line",self.side)
            self.getLineStats(
                curImgFtr.getRoadProjection(), faint=faint, resized=resized)

            # set bottom of line
            x = polynomial([self.projectedY - 1])
            self.pixelBasePos = x[0]

    # create adjacent lane lines using an existing lane on the left
    def createPolyFitRight(self, curImgFtr, leftLane,
                           faint=1.0, resized=False):
        # create new right line polynomial
        polyDiff = np.polysub(leftLane.lines[leftLane.right].currentFit,
                              leftLane.lines[leftLane.left].currentFit)
        self.currentFit = np.polyadd(
            leftLane.lines[leftLane.right].currentFit, polyDiff)
        polynomial = np.poly1d(self.currentFit)
        self.allY = leftLane.lines[leftLane.right].allY
        self.currentX = polynomial(self.allY)
        self.allX = self.currentX

        if len(self.allY) > 75:
            # We need to increase our pixel count by 2 to get to 100%
            # confidence and maintain the current pixel count to keep
            # the line detection
            self.confidence_based = len(self.allY) * 2
            self.confidence = len(self.allY) / self.confidence_based
            self.detected = True

            # create linepoly
            xy1 = np.column_stack(
                (self.currentX + self.maskDelta, self.allY))
            xy1 = xy1.astype(np.int32)
            xy2 = np.column_stack(
                (self.currentX - self.maskDelta, self.allY))
            xy2 = xy2.astype(np.int32)
            self.linePoly = np.concatenate((xy1, xy2[::-1]), axis=0)

            # create mask
            self.linemask = np.zeros_like(self.linemask)
            cv2.fillConvexPoly(self.linemask, self.linePoly, 64)

            # Add the point at the bottom.
            allY = np.append(self.allY, self.projectedY - 1)
            allX = polynomial(allY)
            self.XYPolyline = np.column_stack((allX, allY))
            self.XYPolyline = self.XYPolyline.astype(np.int32)

            # create the accumulator
            self.bestFit = self.currentFit

            # classify the line
            # print("classifying the right line",self.side)
            self.getLineStats(
                curImgFtr.getRoadProjection(), faint=faint, resized=resized)

            # set bottom of line
            x = polynomial([self.projectedY - 1])
            self.pixelBasePos = x[0]

    # update adjacent lane lines using an existing lane on the right
    def updatePolyFitLeft(self, rightLane):
        # update new left line polynomial
        polyDiff = np.polysub(rightLane.lines[rightLane.left].currentFit,
                              rightLane.lines[rightLane.right].currentFit)
        self.currentFit = np.polyadd(
            rightLane.lines[rightLane.left].currentFit, polyDiff)
        polynomial = np.poly1d(self.currentFit)
        self.allY = rightLane.lines[rightLane.left].allY
        self.currentX = polynomial(self.allY)
        self.allX = self.currentX

        if len(self.allY) > 150:
            # We need to increase our pixel count by 2 to get to 100%
            # confidence and maintain the current pixel count to keep
            # the line detection
            self.confidence = len(self.allY) / self.confidence_based
            if self.confidence > 0.5:
                self.detected = True
                if self.confidence > 1.0:
                    self.confidence = 1.0
            else:
                self.detected = False

            # create linepoly
            xy1 = np.column_stack(
                (self.currentX + self.maskDelta, self.allY)).astype(np.int32)
            xy2 = np.column_stack(
                (self.currentX - self.maskDelta, self.allY)).astype(np.int32)
            self.linePoly = np.concatenate((xy1, xy2[::-1]), axis=0)

            # create mask
            self.linemask = np.zeros_like(self.linemask)
            cv2.fillConvexPoly(self.linemask, self.linePoly, 64)

            # Add the point at the bottom.
            allY = np.append(self.allY, self.projectedY - 1)
            allX = polynomial(allY)
            self.XYPolyline = np.column_stack((allX, allY)).astype(np.int32)

            # create the accumulator
            self.bestFit = self.currentFit

    # update adjacent lane lines using an existing lane on the left
    def updatePolyFitRight(self, leftLane):
        # update new right line polynomial
        polyDiff = np.polysub(leftLane.lines[leftLane.right].currentFit,
                              leftLane.lines[leftLane.left].currentFit)
        self.currentFit = np.polyadd(
            leftLane.lines[leftLane.right].currentFit, polyDiff)
        polynomial = np.poly1d(self.currentFit)
        self.allY = leftLane.lines[leftLane.right].allY
        self.currentX = polynomial(self.allY)
        self.allX = self.currentX

        if len(self.allY) > 150:
            # We need to increase our pixel count by 2 to get to 100%
            # confidence and maintain the current pixel count to keep
            # the line detection
            self.confidence = len(self.allY) / self.confidence_based
            if self.confidence > 0.5:
                self.detected = True
                if self.confidence > 1.0:
                    self.confidence = 1.0
            else:
                self.detected = False

            # create linepoly
            xy1 = np.column_stack(
                (self.currentX + self.maskDelta, self.allY)).astype(np.int32)
            xy2 = np.column_stack(
                (self.currentX - self.maskDelta, self.allY)).astype(np.int32)
            self.linePoly = np.concatenate((xy1, xy2[::-1]), axis=0)

            # create mask
            self.linemask = np.zeros_like(self.linemask)
            cv2.fillConvexPoly(self.linemask, self.linePoly, 64)

            # Add the point at the bottom.
            allY = np.append(self.allY, self.projectedY - 1)
            allX = polynomial(allY)
            self.XYPolyline = np.column_stack((allX, allY)).astype(np.int32)

            # create the accumulator
            self.bestFit = self.currentFit

    # function to find bottom of projection (camera cone)
    def findBottomOfLine(self, curImgFtr):
        projection = curImgFtr.getRoadProjection()
        masked_projection = self.applyLineMask(projection)
        points = np.nonzero(masked_projection)
        self.bottomProjectedY = np.max(points[0])

    # function to find lane line positions given histogram row,
    # last column positions and n_neighbors
    # return column positions
    def find_lane_nearest_neighbors(self, histogram, lastpos, nneighbors):
        ncol = len(histogram) - 1
        x = []
        list = {"count": 0, "position": lastpos}
        for i in range(nneighbors):
            if (lastpos + i) < len(histogram) and histogram[lastpos + i] > 0:
                x.append(lastpos + i)
                if list['count'] < histogram[lastpos + i]:
                    list['count'] = histogram[lastpos + i]
                    list['position'] = lastpos + i
            if (lastpos - i) > 0 and histogram[lastpos - i] > 0:
                x.append(lastpos - i)
                if list['count'] < histogram[lastpos - i]:
                    list['count'] = histogram[lastpos - i]
                    list['position'] = lastpos - i
        return list['position'], x

    # function to set base position
    def setBasePos(self, basePos):
        self.pixelBasePos = basePos

    # function to find lane lines points using a sliding window
    # histogram given starting position
    # return arrays x and y positions
    def find_lane_lines_points(self, masked_lines):
        xval = []
        yval = []
        nrows = masked_lines.shape[0] - 1
        neighbors = 12
        pos1 = self.pixelBasePos
        start_row = nrows - 16
        for i in range(int((nrows / neighbors))):
            histogram = np.sum(
                masked_lines[start_row + 10:start_row + 26, :], axis=0)
            histogram = histogram.astype(np.uint8)
            pos2, x = self.find_lane_nearest_neighbors(
                histogram, pos1, int(neighbors * 1.3))
            y = start_row + neighbors
            for i in range(len(x)):
                xval.append(x[i])
                yval.append(y)
            start_row -= neighbors
            pos1 = pos2
        self.allX = np.array(xval)
        self.allY = np.array(yval)

    # scatter plot the points
    def scatter_plot(self, img, size=3):
        if self.side == 1:
            color = (192, 128, 128)
        else:
            color = (128, 128, 192)
        xy_array = np.column_stack((self.allX, self.allY))
        xy_array = xy_array.astype(np.int32)
        for xy in xy_array:
            cv2.circle(img, (xy[0], xy[1]), size, color, -1)

    # draw fitted polyline
    def polyline(self, img, size=5):
        if self.side == 1:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        cv2.polylines(img, [self.XYPolyline], 0, color, size)

    # Fit a (default=second order) polynomial to lane line
    # Use for initialization when first starting up or when lane line was lost
    # and starting over.
    def fitpoly(self, degree=2):
        if len(self.allY) > 150:
            # We need to increase our pixel count by 2 to get to 100%
            # confidence and maintain the current pixel count to keep
            # the line detection
            self.confidence_based = len(self.allY) * 2
            self.confidence = len(self.allY) / self.confidence_based
            self.detected = True

            self.currentFit = np.polyfit(self.allY, self.allX, degree)
            polynomial = np.poly1d(self.currentFit)

            # reverse the polyline since we went up from the bottom
            self.allY = self.allY[::-1]
            self.currentX = polynomial(self.allY)

            # create linepoly
            xy1 = np.column_stack(
                (self.currentX + 15, self.allY)).astype(np.int32)
            xy2 = np.column_stack(
                (self.currentX - 15, self.allY)).astype(np.int32)
            self.linePoly = np.concatenate((xy1, xy2[::-1]), axis=0)

            # create mask
            self.linemask = np.zeros_like(self.linemask)
            cv2.fillConvexPoly(self.linemask, self.linePoly, 64)

            # Add the point at the bottom.
            # NOTE: we walked up from the bottom - so the base point should be
            # the first point.
            allX = polynomial(self.allY)
            allY = np.append(self.allY, self.projectedY - 1)
            allX = np.append(allX, self.pixelBasePos)
            self.XYPolyline = np.column_stack((allX, allY)).astype(np.int32)

            # create the accumulator
            self.bestFit = self.currentFit

    # Fit a (default=second order) polynomial to lane line
    # This version assumes that a manual fit was already done and is using
    # the previously generated poly to fit the current line in the new frame
    def fitpoly2(self, degree=2):
        if len(self.allY) > 50:
            self.currentFit = np.polyfit(self.allY, self.allX, degree)

            # sanity check
            self.diffs = self.currentFit - self.bestFit
            if abs(sum(self.diffs)) < 150.0:
                polynomial = np.poly1d(self.currentFit)

                # Add the point at the bottom.
                # NOTE: these points are counted by numpy:
                # it does topdown, so our bottom point
                # is now at the end of the list.
                x = polynomial([self.projectedY - 1])
                self.allY = np.append(self.allY, self.projectedY - 1)
                self.allX = np.append(self.allX, x[0])
                self.pixelBasePos = x[0]

                # honoring the road manager request to move higher
                # NOTE: these points are counted by numpy:
                # it does topdown, so our top point
                # is now at the front of the list.
                if self.newYTop is not None:
                    x = polynomial([self.newYTop])
                    self.allY = np.insert(self.allY, 0, self.newYTop)
                    self.allX = np.insert(self.allX, 0, x[0])
                    self.newYTop = None

                # fit the poly and generate the current fit.
                self.currentX = polynomial(self.allY)
                self.XYPolyline = np.column_stack(
                    (self.currentX, self.allY)).astype(np.int32)

                # create linepoly
                xy1 = np.column_stack(
                    (self.currentX + self.maskDelta, self.allY))
                xy1 = xy1.astype(np.int32)
                xy2 = np.column_stack(
                    (self.currentX - self.maskDelta, self.allY))
                xy2 = xy2.astype(np.int32)
                self.linePoly = np.concatenate((xy1, xy2[::-1]), axis=0)

                # create mask
                self.linemask = np.zeros_like(self.linemask)
                cv2.fillConvexPoly(self.linemask, self.linePoly, 64)

                # add to the accumulators
                self.bestFit = (self.bestFit + self.currentFit) / 2

                # figure out confidence level
                self.confidence = len(self.allY) / self.confidence_based
                if self.confidence > 0.5:
                    self.detected = True
                    if self.confidence > 1.0:
                        self.confidence = 1.0
                else:
                    self.detected = False
            else:
                # difference check failed - need to re-initialize
                self.confidence = 0.0
                self.detected = False
        else:
            # not enough points - need to re-initialize
            self.confidence = 0.0
            self.detected = False

    # apply the line masking poly
    def applyLineMask(self, img):
        # print("img: ", img.shape)
        # print("self.linemask: ", self.linemask.shape)
        img0 = img[:, :, 1]
        masked_edge = np.copy(self.linemask).astype(np.uint8)
        masked_edge[(masked_edge > 0)] = 255
        return cv2.bitwise_and(img0, img0, mask=masked_edge)

    # apply the reverse line masking poly
    def applyReverseLineMask(self, img):
        # print("img: ", img.shape)
        # print("self.linemask: ", self.linemask.shape)
        masked_edge = np.copy(self.linemask).astype(np.uint8)
        masked_edge[(self.linemask == 0)] = 255
        masked_edge[(masked_edge < 255)] = 0
        return cv2.bitwise_and(img, img, mask=masked_edge)

    # sample line color'
    def getLineStats(self, img, faint=1.0, resized=False):
        # print("side: ", self.side)
        # print("img: ", img.shape)
        # print("self.linemask: ", self.linemask.shape)
        imgR = np.copy(img[:, :, 0])
        imgG = np.copy(img[:, :, 1])
        imgB = np.copy(img[:, :, 2])
        masked_edge = np.copy(self.linemask).astype(np.uint8)
        masked_edge[(masked_edge > 0)] = 255
        imgR = cv2.bitwise_and(imgR, imgR, mask=masked_edge)
        imgG = cv2.bitwise_and(imgG, imgG, mask=masked_edge)
        imgB = cv2.bitwise_and(imgB, imgB, mask=masked_edge)
        red = np.max(imgR)
        green = np.max(imgG)
        blue = np.max(imgB)
        self.line_rgb = (red, green, blue)
        if len(self.allY) > 0:
            self.line_height = np.max(self.allY) - np.min(self.allY)
            self.pixelDensity = len(self.allY) / self.line_height
            if resized:
                self.pixelDensity *= 1.05
            self.lineClassified = True
        else:
            self.line_height = 0
            self.pixelDensity = 0.00000001

        # detect if we have yellow lane line or white lane line
        if (red > np.absolute(200 * faint) and
            green < np.absolute(200 * faint) and
            blue < np.absolute(200 * faint)) or \
            (red > np.absolute(200 * faint) and
             green > np.absolute(200 * faint) and
             blue < np.absolute(200 * faint)):
            if faint == 1.0:
                self.line_color = "yellow"
            elif not resized and np.absolute(faint) < 0.7 and faint > 0.5:
                self.line_color = "white"
                self.lineType = "solid"
                self.lineClassified = True
            else:
                self.line_color = "white"
        elif (red > np.absolute(200 * faint) and
              green > np.absolute(200 * faint) and
              blue > np.absolute(200 * faint)):
            if resized and np.absolute(faint) < 0.8:
                if faint < -0.5:
                    self.line_color = "white"
                    self.lineType = "dashed"
                    self.lineClassified = True
                elif faint < 0.0:
                    self.line_color = "white"
                    self.lineType = "solid"
                    self.lineClassified = True
                else:
                    self.line_color = "yellow"
                    self.lineType = "solid"
                    self.lineClassified = True
            elif not resized and np.absolute(faint) < 0.7:
                if faint < -0.50:
                    self.line_color = "yellow"
                    self.lineType = "solid"
                    self.lineClassified = True
                else:
                    self.line_color = "white"
                    self.lineType = "solid"
                    self.lineClassified = True
            else:
                self.line_color = "white"
        else:
            self.lineClassified = False
            self.line_color = ""

        # detect if we have solid or dashed lines
        if self.pixelDensity > 1.0 or self.line_color == "yellow":
            self.lineType = "solid"
        elif self.pixelDensity < 0.0001:
            self.lineClassified = False
            self.lineType = "not found"
        elif self.lineType == "not found":
            self.lineType = "dashed"

        # determine if we should have adjacent lane lines
        # left solid yellow
        if self.lineType == "solid" and \
           self.line_color == "yellow" and \
           self.side == 1:
            self.adjacentLeft = False
            self.adjacentLLine = None
            self.adjacentRight = True
            self.adjacentRLine = None
        elif (self.lineType == "solid" and
              self.line_color == "white" and
              self.side == 2):
            self.adjacentLeft = True
            self.adjacentLLine = None
            self.adjacentRight = False
            self.adjacentRLine = None
        elif self.line_height > 600:
            self.adjacentLeft = True
            self.adjacentLLine = None
            self.adjacentRight = True
            self.adjacentRLine = None
        else:
            self.adjacentLeft = True
            self.adjacentLLine = None
            self.adjacentRight = False
            self.adjacentRLine = None
        # print("line type", self.lineType, "color",
        #        self.line_color, "classified: ", self.lineClassified)
        # print("rgb:", red, green, blue )
        # print("faint:", faint )
        # print("pixel density:", self.pixelDensity )

    # get the top point of the detected line.
    # use to see if we lost track
    def getTopPoint(self):
        if (self.allY is not None and
                self.currentFit is not None and
                len(self.allY) > 0):
            y = np.min(self.allY)
            polynomial = np.poly1d(self.currentFit)
            x = polynomial([y])
            return(x[0], y)
        else:
            return None

    # road manager request to move the line detection higher
    # otherwise the algorithm is lazy and will lose the entire line.
    def requestTopY(self, newY):
        self.newYTop = newY

    # reset the mask delta for dynamically adjusting masking curve when lines
    # are harder to find.
    def setMaskDelta(self, maskDelta):
        self.maskDelta = maskDelta

    # Define conversions in x and y from pixels space to meters given lane
    # line separation in pixels
    # NOTE: Only do calculation if it make sense - otherwise give previous
    # answer.
    def radius_in_meters(self, throwDistanceInMeters, distance):
        # print("throwDistanceInMeters: ", throwDistanceInMeters)
        if (throwDistanceInMeters > 0.0 and
                self.allY is not None and
                self.currentX is not None and
                len(self.allY) > 0 and
                len(self.currentX) > 0 and
                len(self.allY) == len(self.currentX)):

            ###################################################################
            # Note: We are using 100 instead of 30 here since our throw for the
            #       perspective transform is much longer. We estimate our throw
            #       is 100 meters based on US Highway reecommended guides for
            #       Longitudinal Pavement Markings.
            #       See: http://mutcd.fhwa.dot.gov/htm/2003r1/part3/part3a.htm
            #       Section 3A.05 Widths and Patterns of Longitudinal Pavement
            #       Markings.
            #       Guidance:
            #           Broken lines should consist of 3 m (10 ft) line
            #           segments and 9 m (30 ft) gaps, or dimensions in a
            #           similar ratio of line segments to gaps as appropriate
            #           for traffic speeds and need for delineation.
            #       With new Full HD projected dimensions rotated 90 degrees,
            #       lying on its side (1080, 1920), We are detecting about 8
            #       to 9 sets of dashed line lanes on the right side:
            #           8.5x(3+9)=8.5x12=~102m or just round to 100m AS DEFAULT
            #       We are now calculating the throw Distance based on detected
            #       projection top.
            ###################################################################

            # we have more pixels for Y changed from 720 to 1920
            # meters per pixel in y dimension
            ym_per_pix = throwDistanceInMeters / self.projectedY
            # meteres per pixel in x dimension (at the base)
            xm_per_pix = 3.7 / distance
            #
            # Use the middle point in the distance of the road instead of the
            # base where the car is at
            # NOTE:
            # since we are using a 1920 pixel throw, 8 seems to be the correct
            # divisor now.
            ypoint = self.projectedY / 6
            fit_cr = np.polyfit(self.allY * ym_per_pix,
                                self.currentX * xm_per_pix, 2)
            self.radiusOfCurvature = (
                (1 + (2 * fit_cr[0] * ypoint +
                 fit_cr[1])**2)**1.5) / (2 * fit_cr[0])
        return self.radiusOfCurvature

    # Define conversion in x off center from pixel space to meters given lane
    # line separation in pixels
    def meters_from_center_of_vehicle(self, distance):
        # meteres per pixel in x dimension given lane line separation in pixels
        xm_per_pix = 3.7 / distance
        pixels_off_center = int(self.pixelBasePos - (self.projectedX / 2))
        self.lineBasePos = xm_per_pix * pixels_off_center
        return self.lineBasePos
