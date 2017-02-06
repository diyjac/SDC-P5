#!/usr/bin/python
"""
line.py: version 0.1.0

History:
2017/01/29: coding style phase1:
            reformat to python-guide.org code style
            http://docs.python-guide.org/en/latest/writing/style/
            which uses PEP 8 as a base: http://pep8.org/.
2017/01/21: Initial version converted to a class
"""

import numpy as np
import cv2
import math

# Define a class to receive the characteristics of each lane created by a
# pair of lines that it will keep track.


class Lane():

    def __init__(self, x, y, projectedX, projectedY, maskDelta,
                 lines, left=0, right=1, maskvalue=128):
        # initial setup
        self.curFrame = None

        # our own copy of the lines array
        self.lines = lines

        # dimensions
        self.mid = int(y / 2)
        self.x = x
        self.y = y
        self.projectedX = projectedX
        self.projectedY = projectedY

        # frameNumber
        self.currentFrame = None

        # left lines only
        # left line identifier
        self.left = left

        # left lane stats
        self.leftLineLastTop = None
        self.adjacentLeft = False

        # right lines only
        # right line identifier
        self.right = right

        # right lane stats
        self.rightLineLastTop = None
        self.adjacentRight = False

        # number of points fitted
        self.leftLinePoints = 0
        self.rightLinePoints = 0

        # mask value
        self.maskvalue = maskvalue

    # confidence calculation
    def confidence(self):
        lconf = self.lines[self.left].confidence
        rconf = self.lines[self.right].confidence
        if lconf > rconf:
            return rconf
        return lconf

    # function to set left/right indexes
    def setLineIndex(self, left, right):
        self.left = left
        self.right = right

    # function to get left/right indexes
    def getLineIndex(self):
        return self.left, self.right

    # function to get combined lineBasePos
    def getLineBasePos(self):
        lineBasePos = self.lines[self.left].lineBasePos
        lineBasePos += self.lines[self.right].lineBasePos
        return lineBasePos

    # function to draw lane polygon
    def drawLanePoly(self, roadmask):
        if self.lines[self.right].XYPolyline is not None and \
           self.lines[self.left].XYPolyline is not None:
            roadpoly = np.concatenate(
                (self.lines[self.right].XYPolyline,
                 self.lines[self.left].XYPolyline[::-1]), axis=0)
            cv2.fillConvexPoly(roadmask, roadpoly, self.maskvalue)
        return roadmask

    # function to calculate radius of curvature measurements
    def getRadiusOfCurvature(self):
        if self.lines[self.left].radiusOfCurvature is None or \
           self.lines[self.right].radiusOfCurvature is None:
            if self.lines[self.left].radiusOfCurvature is None:
                if self.lines[self.right].radiusOfCurvature is None:
                    radius = 0.000001
                else:
                    radius = self.lines[
                        self.right].radiusOfCurvature
            else:
                if self.lines[self.right].radiusOfCurvature is None:
                    radius = self.lines[self.left].radiusOfCurvature
                else:
                    radius = self.lines[self.left].radiusOfCurvature
                    radius += self.lines[self.right].radiusOfCurvature
                    radius /= 2.0
        elif self.lines[self.left].radiusOfCurvature > 0.0 and \
                self.lines[self.right].radiusOfCurvature > 0.0:
            radius = self.lines[self.left].radiusOfCurvature
            radius += self.lines[self.right].radiusOfCurvature
            radius /= 2.0
            if self.lines[self.left].radiusOfCurvature > 3000.0:
                roadStraight = True
            elif self.lines[self.right].radiusOfCurvature > 3000.0:
                roadStraight = True
            else:
                roadStraight = False
        elif self.lines[self.left].radiusOfCurvature < 0.0 and \
                self.lines[self.right].radiusOfCurvature < 0.0:
            radius = self.lines[self.left].radiusOfCurvature
            radius += self.lines[self.right].radiusOfCurvature
            radius /= 2.0
            if self.lines[self.left].radiusOfCurvature < -3000.0:
                roadStraight = True
            elif self.lines[self.right].radiusOfCurvature < -3000.0:
                roadStraight = True
            else:
                roadStraight = False
        else:
            radius = 0.000001
            roadStraight = True
        return radius, roadStraight

    # function to set maskDelta
    def setMaskDelta(self, maskDelta):
        self.lines[self.left].setMaskDelta(maskDelta)
        self.lines[self.right].setMaskDelta(maskDelta)

    # function to find starting lane line positions
    def findInitialLines(self, curImgFtr, resized=False):
        self.curImgFtr = curImgFtr

        if self.curFrame is None:
            self.curFrame = 0
        else:
            self.curFrame += 1

        masked_edges = curImgFtr.getEdgeProjection()
        # print("masked_edges: ", masked_edges.shape)
        masked_edge = masked_edges[:, :, 1]
        height = masked_edge.shape[0]
        width = masked_edge.shape[1]

        # get the initial points into the lines
        # height used to be half - but with 1920 pixels - we just need 20%
        # now...
        lefthistogram = np.sum(masked_edge[int(
            height * 0.80):height, 0:int(width * 0.5)], axis=0)
        lefthistogram = lefthistogram.astype(np.float32)
        righthistogram = np.sum(masked_edge[int(
            height * 0.80):height, int(width * 0.5):width], axis=0)
        righthistogram = righthistogram.astype(np.float32)
        self.leftpos = np.argmax(lefthistogram)
        self.rightpos = np.argmax(righthistogram) + int(width / 2)
        self.distance = self.rightpos - self.leftpos

        # set the left and right line's base
        self.lines[self.left].setBasePos(self.leftpos)
        self.lines[self.right].setBasePos(self.rightpos)

        # set their points
        self.lines[self.left].find_lane_lines_points(masked_edge)
        self.lines[self.right].find_lane_lines_points(masked_edge)

        # fit the left line side
        self.lines[self.left].fitpoly()
        self.leftprojection = self.lines[self.left].applyLineMask(masked_edges)
        self.lines[self.left].radius_in_meters(
            self.curImgFtr.throwDistance, self.distance)
        self.lines[self.left].meters_from_center_of_vehicle(self.distance)

        # classify the left line
        if not self.lines[self.left].lineClassified:
            # print("classifying the left line",self.left)
            self.lines[self.left].getLineStats(
                self.curImgFtr.getRoadProjection(), resized=resized)
            self.lines[self.left].adjacentRLine = self.lines[self.right]
            self.adjacentLeft = self.lines[self.left].adjacentLeft
            self.adjacentLLane = None

        # fit the right side
        self.lines[self.right].fitpoly()
        self.rightprojection = self.lines[
            self.right].applyLineMask(masked_edges)
        self.lines[self.right].radius_in_meters(
            self.curImgFtr.throwDistance, self.distance)
        self.lines[self.right].meters_from_center_of_vehicle(self.distance)

        # classify the right line
        if not self.lines[self.right].lineClassified:
            # print("classifying the right line",self.right)
            self.lines[self.right].getLineStats(
                self.curImgFtr.getRoadProjection(), resized=resized)
            self.lines[self.right].adjacentLLine = self.lines[self.left]
            self.adjacentRight = self.lines[self.right].adjacentRight
            self.adjacentRLane = None

        # Update Stats and Top points for next frame.
        self.leftLineLastTop = self.lines[self.left].getTopPoint()
        self.rightLineLastTop = self.lines[self.right].getTopPoint()
        self.leftLinePoints = len(self.lines[self.left].allX)
        self.rightLinePoints = len(self.lines[self.right].allX)

    # function to calculate center x position given y for a lane
    def calculateXCenter(self, y):
        leftPolynomial = np.poly1d(self.lines[self.left].currentFit)
        rightPolynomial = np.poly1d(self.lines[self.right].currentFit)
        return int((rightPolynomial([y]) + leftPolynomial([y])) / 2)

    # function to calculate bottom y for a lane
    def bottomY(self):
        return np.min([
            self.lines[self.left].bottomProjectedY,
            self.lines[self.right].bottomProjectedY])

    # function to mask lane line positions
    # and do lane measurement calculations
    def findExistingLines(self, curImgFtr):
        self.curImgFtr = curImgFtr
        self.curFrame += 1

        masked_edges = curImgFtr.getEdgeProjection()
        # print("masked_edges: ", masked_edges.shape)
        masked_edge = masked_edges[:, :, 1]
        height = masked_edge.shape[0]
        width = masked_edge.shape[1]

        # Left Lane Line Projection setup
        self.leftprojection = self.lines[self.left].applyLineMask(masked_edges)
        leftPoints = np.nonzero(self.leftprojection)
        self.lines[self.left].allX = leftPoints[1]
        self.lines[self.left].allY = leftPoints[0]
        self.lines[self.left].fitpoly2()

        # Right Lane Line Projection setup
        self.rightprojection = self.lines[
            self.right].applyLineMask(masked_edges)
        rightPoints = np.nonzero(self.rightprojection)
        self.lines[self.right].allX = rightPoints[1]
        self.lines[self.right].allY = rightPoints[0]
        self.lines[self.right].fitpoly2()

        # take and calculate some measurements
        self.distance = self.lines[
            self.right].pixelBasePos - self.lines[self.left].pixelBasePos
        self.lines[self.left].radius_in_meters(
            self.curImgFtr.throwDistance, self.distance)
        self.lines[self.left].meters_from_center_of_vehicle(self.distance)
        self.lines[self.right].radius_in_meters(
            self.curImgFtr.throwDistance, self.distance)
        self.lines[self.right].meters_from_center_of_vehicle(self.distance)

        leftTop = self.lines[self.left].getTopPoint()
        rightTop = self.lines[self.right].getTopPoint()

        # Attempt to move up the Lane lines if we missed some predictions
        if self.leftLineLastTop is not None and \
                self.rightLineLastTop is not None:

            # If we are in the harder challenge, our visibility is obscured,
            # so only do this if we are certain that our visibility is good.
            # i.e.: not in the harder challenge!
            if self.curImgFtr.visibility > -30:
                # if either lines differs by greater than 50 pixel vertically
                # we need to request the shorter line to go higher.
                if abs(self.leftLineLastTop[1] -
                       self.rightLineLastTop[1]) > 50:
                    if self.leftLineLastTop[1] > self.rightLineLastTop[1]:
                        self.lines[self.left].requestTopY(
                            self.rightLineLastTop[1])
                    else:
                        self.lines[self.right].requestTopY(
                            self.leftLineLastTop[1])

                # if our lane line has fallen to below our threshold, get it to
                # come back up
                if leftTop is not None and leftTop[1] > self.mid - 100:
                    self.lines[self.left].requestTopY(leftTop[1] - 10)
                if leftTop is not None and \
                   leftTop[1] > self.leftLineLastTop[1]:
                    self.lines[self.left].requestTopY(leftTop[1] - 10)
                if rightTop is not None and rightTop[1] > self.mid - 100:
                    self.lines[self.right].requestTopY(rightTop[1] - 10)
                if rightTop is not None and \
                   rightTop[1] > self.rightLineLastTop[1]:
                    self.lines[self.right].requestTopY(rightTop[1] - 10)

            # visibility poor...
            # harder challenge... need to be less agressive going back
            # up the lane... let at least 30 frame pass before trying to
            # move forward.
            elif self.curFrame > 30:
                # if either lines differs by greater than 50 pixel vertically
                # we need to request the shorter line to go higher.
                if abs(self.leftLineLastTop[1] -
                       self.rightLineLastTop[1]) > 50:
                    if self.leftLineLastTop[1] > self.rightLineLastTop[1] and \
                       leftTop is not None:
                        self.lines[self.left].requestTopY(leftTop[1] - 10)
                    elif rightTop is not None:
                        self.lines[self.right].requestTopY(rightTop[1] - 10)

                # if our lane line has fallen to below our threshold, get it to
                # come back up
                if leftTop is not None and leftTop[1] > self.mid + 100:
                    self.lines[self.left].requestTopY(leftTop[1] - 10)
                if leftTop is not None and \
                   leftTop[1] > self.leftLineLastTop[1]:
                    self.lines[self.left].requestTopY(leftTop[1] - 10)
                if rightTop is not None and rightTop[1] > self.mid + 100:
                    self.lines[self.right].requestTopY(rightTop[1] - 10)
                if rightTop is not None and \
                   rightTop[1] > self.rightLineLastTop[1]:
                    self.lines[self.right].requestTopY(rightTop[1] - 10)

        # Update Stats and Top points for next frame.
        self.leftLineLastTop = self.lines[self.left].getTopPoint()
        self.rightLineLastTop = self.lines[self.right].getTopPoint()
        self.leftLinePoints = len(self.lines[self.left].allX)
        self.rightLinePoints = len(self.lines[self.right].allX)
