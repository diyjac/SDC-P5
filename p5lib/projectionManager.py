#!/usr/bin/python
"""
projectionManager.py: version 0.1.0

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
from p5lib.cameraCal import CameraCal
from p5lib.imageFilters import ImageFilters


class ProjectionManager():

    # Initialize projectionManager
    def __init__(self, camCal, keepN=10, gradientLevel=75, debug=False):
        # set debugging
        self.debug = debug

        # frameNumber
        self.curFrame = None

        # keep last N
        self.keepN = keepN

        # keep our own copy of the camera calibration
        self.camCal = camCal

        # our own copy of the camera calibration results
        self.mtx, self.dist, self.img_size = camCal.get()

        # normal image size
        self.x, self.y = self.img_size
        # based on hough3 (default)
        self.z = self.y/45

        # projection mask calculations
        self.xbottom1 = int(self.x / 16)
        self.xbottom2 = int(self.x * 15 / 16)
        self.xtop1 = int(self.x * 14 / 32)
        self.xtop2 = int(self.x * 18 / 32)
        self.ybottom1 = self.y
        self.ybottom2 = self.y
        self.ytopbox = int(self.y * 9 / 16)

        # mid point in picture (by height)
        self.mid = int(self.y / 2)

        # ghosting
        self.roadGhost = np.zeros((self.mid, self.x), dtype=np.uint8)

        # gradient level starts here
        self.gradient0 = self.mid + gradientLevel

        # current image Filter
        self.curImgFtr = None

        # current road corners
        self.curSrcRoadCorners = None

        # current horizon
        self.curHorizon = None

        # current gradient
        self.curGradient = None

        # last n projected image filters
        self.recentProjected = []

        # last n road corners
        self.recentRoadCorners = []

        # last n horizon detected
        self.recentHorizon = []

        # last n gradient detected
        self.recentGradient = []

        # for 3D reconstruction and augmentation
        self.rvecs = None
        self.tvecs = None
        self.inliers = None

        # our projection settings - FULLHD 1080p on its side.
        self.projectedX = 1080
        self.projectedY = 1920

        # set up debugging diag screens
        if self.debug:
            self.diag1 = np.zeros((self.mid, self.x, 3), dtype=np.float32)
            self.diag2 = np.zeros((self.y, self.x, 3), dtype=np.float32)
            self.diag3 = np.zeros(
                (self.projectedY, self.projectedX, 3), dtype=np.float32)
            self.diag4 = np.zeros(
                (self.projectedY, self.projectedX, 3), dtype=np.float32)

    # set current image filter
    def set_image_filter(self, imgFtr):
        self.curImgFtr = imgFtr

    # create a region of interest mask
    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with
        # depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill
        # color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    # draw outline of given area
    def draw_area_of_interest(self, img, areas,
                              color=[128, 0, 128], thickness=2):
        for points in areas:
            for i in range(len(points) - 1):
                cv2.line(img,
                         (points[i][0],
                          points[i][1]),
                         (points[i + 1][0],
                          points[i + 1][1]), color, thickness)
            cv2.line(img,
                     (points[0][0],
                      points[0][1]),
                     (points[len(points) - 1][0],
                      points[len(points) - 1][1]), color, thickness)

    # draw outline of given area
    def draw_area_of_interest_for_projection(self, img, areas,
                                             color=[128, 0, 128],
                                             thickness1=2,
                                             thickness2=10):
        for points in areas:
            for i in range(len(points) - 1):
                if i == 0 or i == 1:
                    cv2.line(img, (points[i][0], points[i][1]), (points[
                             i + 1][0], points[i + 1][1]), color, thickness1)
                else:
                    cv2.line(img, (points[i][0], points[i][1]), (points[
                             i + 1][0], points[i + 1][1]), color, thickness2)
            cv2.line(img,
                     (points[0][0],
                      points[0][1]),
                     (points[len(points) - 1][0],
                      points[len(points) - 1][1]), color, thickness1)

    def draw_masked_area(self, img, areas, color=[128, 0, 128], thickness=2):
        for points in areas:
            for i in range(len(points) - 1):
                cv2.line(img,
                         (points[i][0],
                          points[i][1]),
                         (points[i + 1][0],
                          points[i + 1][1]), color, thickness)
            cv2.line(img,
                     (points[0][0],
                      points[0][1]),
                     (points[len(points) - 1][0],
                      points[len(points) - 1][1]), color, thickness)

    def draw_bounding_box(self, img, boundingbox,
                          color=[0, 255, 0], thickness=6):
        x1, y1, x2, y2 = boundingbox
        cv2.line(img, (x1, y1), (x2, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y2), color, thickness)
        cv2.line(img, (x2, y2), (x1, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y1), color, thickness)

    # draw parallel lines in a perspective image that will later be projected
    # into a flat surface
    def draw_parallel_lines_pre_projection(self, img, lane_info,
                                           color=[128, 0, 0], thickness=5):
        lx1 = lane_info[3][0]
        rx1 = lane_info[4][0]
        rx2 = lane_info[5][0]
        lx2 = lane_info[6][0]
        ly1 = lane_info[3][1]
        ry1 = lane_info[4][1]
        ry2 = lane_info[5][1]
        ly2 = lane_info[6][1]
        cv2.line(img, (lx1, ly1), (lx2, ly2), color, thickness)
        cv2.line(img, (rx1, ry1), (rx2, ry2), color, thickness)

    def draw_estimated_lane_line_location(self, img, base_pos, distance,
                                          color=[128, 0, 0], thickness=5):
        x = int(base_pos + distance)
        y1 = self.projectedY - 750
        y2 = self.projectedY
        cv2.line(img, (x, y1), (x, y2), color, thickness)

    # calculate and draw initial estimated lines on the roadway.
    def draw_lines(self, img, lines,
                   color=[255, 0, 0], thickness=6, backoff=0, debug=False):
        if backoff == 0:
            backoff = thickness * 2
            # backoff=thickness*5
        ysize = img.shape[0]
        midleft = img.shape[1] / 2 - 200 + backoff * 2
        midright = img.shape[1] / 2 + 200 - backoff * 2
        top = ysize / 2 + backoff * 2
        rightslopemin = 0.5  # 8/backoff
        rightslopemax = 3.0  # backoff/30
        leftslopemax = -0.5  # -8/backoff
        leftslopemin = -3.0  # -backoff/30
        try:
            # rightline and leftline cumlators
            rl = {'num': 0, 'slope': 0.0, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
            ll = {'num': 0, 'slope': 0.0, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
            for line in lines:
                for x1, y1, x2, y2 in line:
                    slope = ((y2 - y1) / (x2 - x1))
                    sides = (x1 + x2) / 2
                    vmid = (y1 + y2) / 2
                    if (slope > rightslopemin and slope < rightslopemax and
                            sides > midright and vmid > top):   # right
                        if debug:
                            # print("x1,y1,x2,y2: ", x1, y1, x2, y2)
                            cv2.line(img, (x1, y1), (x2, y2),
                                     [128, 128, 0], thickness)
                        rl['num'] += 1
                        rl['slope'] += slope
                        rl['x1'] += x1
                        rl['y1'] += y1
                        rl['x2'] += x2
                        rl['y2'] += y2
                    elif (slope > leftslopemin and slope < leftslopemax and
                          sides < midleft and vmid > top):   # left
                        if debug:
                            # print("x1,y1,x2,y2: ", x1, y1, x2, y2)
                            cv2.line(img, (x1, y1), (x2, y2),
                                     [128, 128, 0], thickness)
                        ll['num'] += 1
                        ll['slope'] += slope
                        ll['x1'] += x1
                        ll['y1'] += y1
                        ll['x2'] += x2
                        ll['y2'] += y2

            if rl['num'] > 0 and ll['num'] > 0:
                # average/extrapolate all of the lines that makes the right
                # line
                rslope = rl['slope'] / rl['num']
                rx1 = int(rl['x1'] / rl['num'])
                ry1 = int(rl['y1'] / rl['num'])
                rx2 = int(rl['x2'] / rl['num'])
                ry2 = int(rl['y2'] / rl['num'])

                # average/extrapolate all of the lines that makes the left line
                lslope = ll['slope'] / ll['num']
                lx1 = int(ll['x1'] / ll['num'])
                ly1 = int(ll['y1'] / ll['num'])
                lx2 = int(ll['x2'] / ll['num'])
                ly2 = int(ll['y2'] / ll['num'])

                # find the right and left line's intercept, which means solving
                # the following two equations:
                #
                # rslope = ( yi - ry1 )/( xi - rx1)
                # lslope = ( yi = ly1 )/( xi - lx1)
                # solve for (xi, yi): the intercept of the left and right lines
                # which is:
                #   xi = (ly2 - ry2 + rslope*rx2 - lslope*lx2)/(rslope-lslope)
                # and
                #   yi = ry2 + rslope*(xi-rx2)
                xi = int((ly2 - ry2 + rslope * rx2 -
                          lslope * lx2) / (rslope - lslope))
                yi = int(ry2 + rslope * (xi - rx2))

                # calculate backoff from intercept for right line
                if (rslope > rightslopemin and
                        rslope < rightslopemax):   # right
                    ry1 = yi + int(backoff)
                    rx1 = int(rx2 - (ry2 - ry1) / rslope)
                    ry2 = ysize - 1
                    rx2 = int(rx1 + (ry2 - ry1) / rslope)
                    cv2.line(img, (rx1, ry1), (rx2, ry2),
                             [255, 0, 0], thickness)

                # calculate backoff from intercept for left line
                if (lslope < leftslopemax and
                        lslope > leftslopemin):   # left
                    ly1 = yi + int(backoff)
                    lx1 = int(lx2 - (ly2 - ly1) / lslope)
                    ly2 = ysize - 1
                    lx2 = int(lx1 + (ly2 - ly1) / lslope)
                    cv2.line(img, (lx1, ly1), (lx2, ly2),
                             [255, 0, 0], thickness)

                # if we have all of the points - draw the backoff line near the
                # horizon
                if lx1 > 0 and ly1 > 0 and rx1 > 0 and ry1 > 0:
                    cv2.line(img, (lx1, ly1), (rx1, ry1),
                             [255, 0, 0], thickness)

            # return the left and right line slope, found rectangler box shape
            # and the estimated vanishing point.
            return lslope + rslope, lslope, rslope, \
                (lx1, ly1), (rx1, ry1), (rx2, ry2), (lx2, ly2), (xi, yi)
        except:
            return -1000, 0.0, 0.0, (0, 0), (0, 0), (0, 0), (0, 0)

    # generate a set of hough lines and calculates its estimates for lane lines
    def hough_lines(self, img, rho, theta, threshold,
                    min_line_len, max_line_gap, backoff=0, debug=False):
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn using the new single line
            for left and right lane line method.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
            []), minLineLength=min_line_len, maxLineGap=max_line_gap)
        masked_lines = np.zeros(img.shape, dtype=np.uint8)
        lane_info = self.draw_lines(
            masked_lines, lines, backoff=backoff, debug=debug)

        return masked_lines, lane_info

    # function to project the undistorted camera image to a plane looking down.
    def unwarp_lane(self, img, src, dst, mtx):
        # Pass in your image, 4 source points:
        #     src = np.float32([[,],[,],[,],[,]])
        # and 4 destination points:
        #     dst = np.float32([[,],[,],[,],[,]])
        # Note: you could pick any four of the detected corners
        # as long as those four corners define a rectangle
        # One especially smart way to do this would be to use four well-chosen
        # use cv2.getPerspectiveTransform() to get M, the transform matrix
        # use cv2.warpPerspective() to warp your image to a top-down view

        self.src2dstM = cv2.getPerspectiveTransform(src, dst)
        self.dst2srcM = cv2.getPerspectiveTransform(dst, src)
        img_size = (self.projectedX, self.projectedY)
        warped = cv2.warpPerspective(
            img, self.src2dstM, img_size, flags=cv2.INTER_LINEAR)

        # warped = gray
        return warped, self.src2dstM

    # function to project the undistorted camera image to a plane looking down.
    def unwarp_lane_back(self, img, src, dst, mtx):
        # Pass in your image, 4 source points:
        #     src = np.float32([[,],[,],[,],[,]])
        # and 4 destination points:
        #     dst = np.float32([[,],[,],[,],[,]])
        # Note: you could pick any four of the detected corners
        # as long as those four corners define a rectangle
        # One especially smart way to do this would be to use four well-chosen
        # use cv2.getPerspectiveTransform() to get M, the transform matrix
        # use cv2.warpPerspective() to warp your image to a top-down view

        img_size = (self.x, self.y)
        warped = cv2.warpPerspective(
            img, self.dst2srcM, img_size, flags=cv2.INTER_LINEAR)

        # warped = gray
        return warped, self.dst2srcM

    # function to find starting lane line positions
    # return left and right column positions
    def find_lane_locations(self, masked_lines):
        height = masked_lines.shape[0]
        width = masked_lines.shape[1]
        lefthistogram = np.sum(
            masked_lines[int(height / 2):height, 0:int(width / 2)],
            axis=0).astype(np.float32)
        righthistogram = np.sum(
            masked_lines[int(height / 2):height, int(width / 2):width],
            axis=0).astype(np.float32)
        leftpos = np.argmax(lefthistogram)
        rightpos = np.argmax(righthistogram) + int(width / 2)
        # print("leftpos",leftpos,"rightpos",rightpos)
        return leftpos, rightpos, rightpos - leftpos

    # hough version1
    def hough_lines1(self, masked_edges, debug=False):
        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 2  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        # minimum number of votes (intersections in Hough grid cell)
        threshold = 40
        # 50 75 25 minimum number of pixels making up a line
        min_line_length = 120
        # 40 50 20 maximum gap in pixels between connectable line segments
        max_line_gap = 40
        return self.hough_lines(masked_edges, rho, theta, threshold,
                                min_line_length, max_line_gap,
                                backoff=30, debug=debug)

    # hough version2
    def hough_lines2(self, masked_edges, debug=False):
        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 2  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        # minimum number of votes (intersections in Hough grid cell)
        threshold = 40
        # 50 75 25 minimum number of pixels making up a line
        min_line_length = 100
        # 40 50 20 maximum gap in pixels between connectable line segments
        max_line_gap = 40
        return self.hough_lines(masked_edges, rho, theta, threshold,
                                min_line_length, max_line_gap,
                                backoff=40, debug=debug)

    # hough version3
    def hough_lines3(self, masked_edges, debug=False):
        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 2  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        # minimum number of votes (intersections in Hough grid cell)
        threshold = 40
        # 50 75 25 minimum number of pixels making up a line
        min_line_length = 75
        # 40 50 20 maximum gap in pixels between connectable line segments
        max_line_gap = 40
        return self.hough_lines(masked_edges, rho, theta, threshold,
                                min_line_length, max_line_gap,
                                backoff=40, debug=debug)

    # hough version4
    def hough_lines4(self, masked_edges, debug=False):
        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 2  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        # minimum number of votes (intersections in Hough grid cell)
        threshold = 40
        # 50 75 25 minimum number of pixels making up a line
        min_line_length = 50
        # 40 50 20 maximum gap in pixels between connectable line segments
        max_line_gap = 30
        return self.hough_lines(masked_edges, rho, theta, threshold,
                                min_line_length, max_line_gap,
                                backoff=50, debug=debug)

    # hough version5
    def hough_lines5(self, masked_edges, debug=False):
        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 2  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        # minimum number of votes (intersections in Hough grid cell)
        threshold = 40
        # 50 75 25 minimum number of pixels making up a line
        min_line_length = 20
        # 40 50 20 maximum gap in pixels between connectable line segments
        max_line_gap = 20
        return self.hough_lines(masked_edges, rho, theta, threshold,
                                min_line_length, max_line_gap,
                                backoff=50, debug=debug)

    # function to find initial road corners to find a projection matrix.
    # once corners are found, project the edges into a plane
    # or when we fall below 50% confidence in the lane line detected.
    def findInitialRoadCorners(self, imgftr):
        # first time?
        if self.curFrame is None:
            self.curFrame = 0
        else:
            self.curFrame += 1

        # piece together images that we want to project
        edge = imgftr.edges()[:, :, 0]

        # We are defining a four sided polygon to mask
        vertices = np.array([[(self.xbottom1, self.ybottom1),
                              (self.xtop1, self.ytopbox),
                              (self.xtop2, self.ytopbox),
                              (self.xbottom2, self.ybottom2)]], dtype=np.int32)

        # now mask it
        masked_edge = self.region_of_interest(np.copy(edge), vertices)
        masked_edges = np.dstack((edge, masked_edge, masked_edge))

        # cascading hough mapping line attempts
        self.hough = 1
        line_image, lane_info = self.hough_lines1(masked_edge)
        if lane_info[0] == -1000:
            self.hough = 2
            line_image, lane_info = self.hough_lines2(masked_edge)
            if lane_info[0] == -1000:
                self.hough = 3
                line_image, lane_info = self.hough_lines3(masked_edge)
                if lane_info[0] == -1000:
                    self.hough = 4
                    line_image, lane_info = self.hough_lines4(masked_edge)
                    if lane_info[0] == -1000:
                        self.hough = 5
                        line_image, lane_info = self.hough_lines5(masked_edge)

        # if we made it: calculate the area of interest
        if lane_info[0] > -1000:
            self.curGradient = lane_info[3][1]

            areaOfInterest = np.array([[(lane_info[3][0] - 50,
                                         lane_info[3][1] - 11),
                                        (lane_info[4][0] + 50,
                                         lane_info[4][1] - 11),
                                        (lane_info[4][0] + 525,
                                         lane_info[4][1] + 75),
                                        (lane_info[4][0] +
                                         500, lane_info[5][1]),
                                        (lane_info[4][0] -
                                         500, lane_info[6][1]),
                                        (lane_info[3][0] - 525,
                                         lane_info[3][1] + 75)]],
                                      dtype=np.int32)

            # generate src rect for projection of road to flat plane
            self.curSrcRoadCorners = np.float32(
                [lane_info[3], lane_info[4], lane_info[5], lane_info[6]])

            # generate destination rect for projection of road to flat plane
            us_lane_width = 12     # US highway width: 12 feet wide
            if self.hough == 1:
                # slight curvy road - poor visibility (challenge)
                # 30.0 Approximate distance to vanishing point
                # from end of rectangle
                approx_dest = 30.0
                self.z = self.y/40
            elif self.hough == 2:
                # road is almost straight - high visibility
                # 42.0 Approximate distance to vanishing point
                # from end of rectangle
                approx_dest = 42.0
                self.z = self.y/55
            elif self.hough == 3:
                # slightly more curvy road - normal visibility
                # 36.0 # 35.56 Approximate distance to vanishing point from
                # end of rectangle
                approx_dest = 36.0
                self.z = self.y/45
            elif self.hough == 4:
                # curvy road - lower visibility
                # 25.0 Approximate distance to vanishing point
                # from end of rectangle
                approx_dest = 25.0
                self.z = self.y/35
            else:
                # very curvy road - very low visibility (harder challenge)
                # 20.0 Approximate distance to vanishing point
                # from end of rectangle
                approx_dest = 15.0
                self.z = self.y/20

            scale_factor = 6.0     # scaling for display
            top = approx_dest * approx_dest
            left = -(us_lane_width / 2) * scale_factor
            right = (us_lane_width / 2) * scale_factor
            self.curDstRoadCorners = np.float32(
                [[(self.projectedX / 2) + left, top],
                 [(self.projectedX / 2) + right, top],
                 [(self.projectedX / 2) + right,
                  self.projectedY],
                 [(self.projectedX / 2) + left,
                  self.projectedY]])

            # create 3D dst road corners
            self.cur3DDstRoadCorners = np.zeros((4, 3), np.float32)
            self.cur3DDstRoadCorners[:, :2] = self.curDstRoadCorners

            # generate grayscaled map image for projection
            projected_roadsurface, M = self.unwarp_lane(
                np.copy(masked_edges), self.curSrcRoadCorners,
                self.curDstRoadCorners, self.mtx)
            imgftr.setEdgeProjection(projected_roadsurface)

            # generate full color projection
            projected, M = self.unwarp_lane(
                imgftr.curImage, self.curSrcRoadCorners,
                self.curDstRoadCorners, self.mtx)
            imgftr.setRoadProjection(projected)

            # save current source projection rect.
            self.lane_info = lane_info

            # calculate the rotation and translation vectors for augmentation
            # ret, self.rvecs, self.tvecs, inliners = cv2.solvePnPRansac(
            #     self.cur3DDstRoadCorners.reshape([4,1,3]),
            #     self.curSrcRoadCorners.reshape([4,1,2]),
            #     self.mtx, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
            # ret, self.rvecs, self.tvecs = cv2.solvePnP(
            #     self.cur3DDstRoadCorners.reshape([4,1,3]),
            #     self.curSrcRoadCorners.reshape([4,1,2]),
            #     self.mtx, self.dist, self.rvecs, self.tvecs,
            #     useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            ret, self.rvecs, self.tvecs = cv2.solvePnP(
                self.cur3DDstRoadCorners.reshape([4, 1, 3]),
                self.curSrcRoadCorners.reshape([4, 1, 2]),
                self.mtx, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)

        # create debug/diag screens if required
        if self.debug:
            # diag 1 screen - road edges with masked out area shown
            self.diag1 = imgftr.makehalf(masked_edges) * 4

            # rest is only valid if we are able to get lane_info...
            if lane_info[0] > -1000:
                leftbound = int(lane_info[7][0] - (self.x * 0.1))
                rightbound = int(lane_info[7][0] + (self.x * 0.1))
                topbound = int(lane_info[7][1] - (self.y * 0.15))
                bottombound = int(lane_info[7][1] + (self.y * 0.05))
                boundingbox = (leftbound - 2, topbound - 2,
                               rightbound + 2, bottombound + 2)

                # non-projected image with found points
                ignore = np.copy(line_image) * 0
                self.diag2 = imgftr.miximg(imgftr.curImage, masked_edges * 2)
                self.diag2 = imgftr.miximg(
                    self.diag2, np.dstack((line_image, ignore, ignore)))
                if imgftr.visibility > -30:
                    self.draw_masked_area(self.diag2, vertices)
                # self.draw_bounding_box(self.diag2, boundingbox)
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(self.diag2, 'Frame: %d   Hough: %d' % (
                    self.curFrame, self.hough),
                    (30, 30), font, 1, (255, 0, 0), 2)
                self.draw_area_of_interest(self.diag2, areaOfInterest, color=[
                                           0, 128, 0], thickness=5)

                cv2.putText(
                    self.diag2, 'x1,y1: %d,%d' %
                    (int(lane_info[3][0]), int(lane_info[3][1])),
                    (int(lane_info[3][0]) - 250, int(lane_info[3][1]) - 30),
                    font, 1, (255, 0, 0), 2)
                cv2.putText(
                    self.diag2, 'x2,y2: %d,%d' %
                    (int(lane_info[4][0]), int(lane_info[4][1])),
                    (int(lane_info[4][0]), int(lane_info[4][1]) - 30),
                    font, 1, (255, 0, 0), 2)
                cv2.putText(
                    self.diag2, 'x3,y3: %d,%d' %
                    (int(lane_info[5][0]), int(lane_info[5][1])),
                    (int(lane_info[5][0]) - 200, int(lane_info[5][1]) - 30),
                    font, 1, (255, 0, 0), 2)
                cv2.putText(
                    self.diag2, 'x4,y4: %d,%d' %
                    (int(lane_info[6][0]), int(lane_info[6][1])),
                    (int(lane_info[6][0]) - 200, int(lane_info[6][1]) - 30),
                    font, 1, (255, 0, 0), 2)

                # diag 3 screen - complete road RGB image projected
                # new_edge = np.copy(edge)
                # new_edge[masked_edge>0] = 0
                # diag3tmp = np.dstack((new_edge, new_edge, new_edge)) * 4
                # diag3tmp = imgftr.miximg(imgftr.curImage, masked_edges*4)
                # self.draw_area_of_interest_for_projection(diag3tmp,
                #     areaOfInterest, color=[0,128,0],
                #     thickness1=1, thickness2=50)
                # self.draw_parallel_lines_pre_projection(diag3tmp,
                #     lane_info, color=[128,0,0], thickness=2)
                self.diag3 = np.copy(projected)

                # diag 4 screen - road edges with masked out area shown
                # projected
                self.diag4, M = self.unwarp_lane(
                    imgftr.makefull(self.diag1),
                    self.curSrcRoadCorners,
                    self.curDstRoadCorners, self.mtx)
                cv2.putText(
                    self.diag4, 'x1,y1: %d,%d' %
                    (int(self.curDstRoadCorners[0][0]),
                     int(self.curDstRoadCorners[0][1]) - 1),
                    (int(self.curDstRoadCorners[0][0]) - 275,
                     int(self.curDstRoadCorners[0][1]) - 15),
                    font, 1, (255, 0, 0), 2)
                cv2.putText(
                    self.diag4, 'x2,y2: %d,%d' %
                    (int(self.curDstRoadCorners[1][0]),
                     int(self.curDstRoadCorners[1][1]) - 1),
                    (int(self.curDstRoadCorners[1][0]) + 25,
                     int(self.curDstRoadCorners[1][1]) - 15),
                    font, 1, (255, 0, 0), 2)
                cv2.putText(
                    self.diag4, 'x3,y3: %d,%d' %
                    (int(self.curDstRoadCorners[2][0]),
                     int(self.curDstRoadCorners[2][1]) - 1),
                    (int(self.curDstRoadCorners[2][0]) + 25,
                     int(self.curDstRoadCorners[2][1]) - 15),
                    font, 1, (255, 0, 0), 2)
                cv2.putText(
                    self.diag4, 'x4,y4: %d,%d' %
                    (int(self.curDstRoadCorners[3][0]),
                     int(self.curDstRoadCorners[3][1]) - 1),
                    (int(self.curDstRoadCorners[3][0]) - 275,
                     int(self.curDstRoadCorners[3][1]) - 15),
                    font, 1, (255, 0, 0), 2)

                # draw circles of destination points
                cv2.circle(
                    self.diag4,
                    (int(self.curDstRoadCorners[0][0]),
                     int(self.curDstRoadCorners[0][1])),
                    10, (255, 64, 64), 10)
                cv2.circle(
                    self.diag4,
                    (int(self.curDstRoadCorners[1][0]),
                     int(self.curDstRoadCorners[1][1])),
                    10, (255, 64, 64), 10)
                cv2.circle(
                    self.diag4,
                    (int(self.curDstRoadCorners[2][0]),
                     int(self.curDstRoadCorners[2][1])),
                    10, (255, 64, 64), 10)
                cv2.circle(
                    self.diag4,
                    (int(self.curDstRoadCorners[3][0]),
                     int(self.curDstRoadCorners[3][1])),
                    10, (255, 64, 64), 10)

    # function to project the edges into a plane
    # this function is for when we are now at greater than 50% confidence in
    # the lane lines identified.
    def project(self, imgftr, leftRightOffset=0, sameFrame=False):
        if not sameFrame:
            self.curFrame += 1

        lane_info = self.lane_info

        # piece together images that we want to project
        edge = imgftr.edges()[:, :, 0]

        # We are defining a four sided polygon to mask
        vertices = np.array([[(self.xbottom1, self.ybottom1),
                              (self.xtop1, self.ytopbox),
                              (self.xtop2, self.ytopbox),
                              (self.xbottom2, self.ybottom2)]],
                            dtype=np.int32)

        # now mask it
        masked_edge = self.region_of_interest(np.copy(edge), vertices)
        masked_edges = np.dstack((edge, masked_edge, masked_edge))

        # calculate the area of interest
        self.curGradient = lane_info[3][1]
        areaOfInterest = np.array(
            [[(lane_info[3][0] - 50 + leftRightOffset,
               lane_info[3][1] - 11),
              (lane_info[4][0] + 50 + leftRightOffset,
               lane_info[4][1] - 11),
              (lane_info[4][0] + 525,
               lane_info[4][1] + 75),
              (lane_info[4][0] + 500, lane_info[5][1]),
              (lane_info[4][0] - 500, lane_info[6][1]),
              (lane_info[3][0] - 525, lane_info[3][1] + 75)]], dtype=np.int32)

        # generate src rect for projection of road to flat plane
        # since this is a fast version of the projector in general
        # we will use the last projection information what we obtained
        # from the last search; however, roadmanager can reset this based
        # on gap thresholds with the last detected horizon.
        # see setSrcTop() function below.
        self.curSrcRoadCorners = np.float32(
            [lane_info[3], lane_info[4], lane_info[5], lane_info[6]])

        # generate grayscaled map image
        projected_roadsurface, M = self.unwarp_lane(
            np.copy(masked_edges), self.curSrcRoadCorners,
            self.curDstRoadCorners, self.mtx)
        imgftr.setEdgeProjection(projected_roadsurface)

        # generate full color projection
        projected, M = self.unwarp_lane(
            imgftr.curImage, self.curSrcRoadCorners,
            self.curDstRoadCorners, self.mtx)
        imgftr.setRoadProjection(projected)

        # re-calculate the rotation and translation vectors for augmentation
        # ret, self.rvecs, self.tvecs, inliners = cv2.solvePnPRansac(
        #     self.cur3DDstRoadCorners.reshape([4,1,3]),
        #     self.curSrcRoadCorners.reshape([4,1,2]),
        #     self.mtx, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
        # ret, self.rvecs, self.tvecs = cv2.solvePnP(
        #     self.cur3DDstRoadCorners.reshape([4,1,3]),
        #     self.curSrcRoadCorners.reshape([4,1,2]),
        #     self.mtx, self.dist, self.rvecs, self.tvecs,
        #     useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        ret, self.rvecs, self.tvecs = cv2.solvePnP(
            self.cur3DDstRoadCorners.reshape([4, 1, 3]),
            self.curSrcRoadCorners.reshape([4, 1, 2]),
            self.mtx, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)

        # create debug/diag screens if required
        if self.debug:
            # diag 1 screen - road edges with masked out area shown
            self.diag1 = imgftr.makehalf(masked_edges) * 4

            # rest is only valid if we are able to get lane_info...
            if lane_info[0] > -1000:
                leftbound = int(lane_info[3][0] - 50 + leftRightOffset)
                rightbound = int(lane_info[4][0] + 50 + leftRightOffset)
                topbound = int(lane_info[3][1] - 71)
                bottombound = int(lane_info[3][1] - 11)
                boundingbox = (leftbound - 2, topbound - 2,
                               rightbound + 2, bottombound + 2)

                # non-projected image with found points
                self.diag2 = imgftr.miximg(imgftr.curImage, masked_edges * 2)
                if imgftr.visibility > -30:
                    self.draw_masked_area(self.diag2, vertices)
                # self.draw_bounding_box(self.diag2, boundingbox)
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    self.diag2, 'Frame: %d   Hough: %d' %
                    (self.curFrame, self.hough),
                    (30, 30), font, 1, (255, 0, 0), 2)
                self.draw_area_of_interest(self.diag2, areaOfInterest, color=[
                                           0, 128, 0], thickness=5)

                # diag 3 screen - complete road RGB image projected
                diag3tmp = imgftr.miximg(imgftr.curImage, masked_edges * 4)
                self.draw_area_of_interest_for_projection(
                    diag3tmp, areaOfInterest, color=[0, 128, 0],
                    thickness1=1, thickness2=50)
                self.draw_parallel_lines_pre_projection(
                    diag3tmp, lane_info, color=[128, 0, 0], thickness=2)
                self.diag3, M = self.unwarp_lane(
                    diag3tmp, self.curSrcRoadCorners,
                    self.curDstRoadCorners, self.mtx)

                # diag 4 screen - road edges with masked out area shown
                # projected
                self.diag4, M = self.unwarp_lane(
                    imgftr.makefull(self.diag1),
                    self.curSrcRoadCorners,
                    self.curDstRoadCorners, self.mtx)

    # warp the perspective view to planar view
    def curWarp(self, imgftr, image):
        warped, M = self.unwarp_lane(
            image, self.curSrcRoadCorners, self.curDstRoadCorners, self.mtx)
        return warped

    # unwarp the planar view back to perspective view
    def curUnWarp(self, imgftr, image):
        unwarped, M = self.unwarp_lane_back(
            image, self.curDstRoadCorners, self.curSrcRoadCorners, self.mtx)
        return unwarped

    # an attempt to dampen the bounce of the car and the road surface.
    # called by RoadManager class
    def setSrcTop(self, newTop, sideDelta):
        if newTop > self.gradient0:
            self.ytopbox = newTop - 15
            self.xtop1 += sideDelta
            self.xtop2 -= sideDelta
            self.lane_info = (self.lane_info[0],
                              self.lane_info[1],
                              self.lane_info[2],
                              (self.lane_info[3][0] + sideDelta, newTop),
                              (self.lane_info[4][0] - sideDelta, newTop),
                              self.lane_info[5],
                              self.lane_info[6],
                              self.lane_info[7])

    # another attempt to dampen the bounce of the car and the road surface.
    def setSrcTopX(self, sideDelta):
        self.lane_info = (self.lane_info[0],
                          self.lane_info[1],
                          self.lane_info[2],
                          (self.xtop1 + sideDelta, self.lane_info[3][1]),
                          (self.xtop2 + sideDelta, self.lane_info[4][1]),
                          self.lane_info[5],
                          self.lane_info[6],
                          (self.lane_info[7][0] + sideDelta,
                           self.lane_info[7][1]))

    # another attempt to dampen the bounce of the car and the road surface.
    # This time by detecting the dest projection top and reset it if it is too
    # low
    def resetDestTop(self, projectionTopPixel):
        us_lane_width = 12     # US highway width: 12 feet wide

        # Approximate distance to vanishing point from end of rectangle
        # calculate back off from vanishing point to move the display up
        approx_dest = 42.0 * \
            (1.0 - ((projectionTopPixel / self.projectedY) * 0.50))

        scale_factor = 6.0     # scaling for display
        top = approx_dest * approx_dest
        left = -(us_lane_width / 2) * scale_factor
        right = (us_lane_width / 2) * scale_factor
        self.curDstRoadCorners = np.float32(
            [[(self.projectedX / 2) + left, top],
             [(self.projectedX / 2) + right, top],
             [(self.projectedX / 2) + right, self.projectedY],
             [(self.projectedX / 2) + left, self.projectedY]])

        # create 3D dst road corners
        self.cur3DDstRoadCorners = np.zeros((4, 3), np.float32)
        self.cur3DDstRoadCorners[:, :2] = self.curDstRoadCorners

    # pixel to meter distance calculation
    def pixel2Meter(self):
        return self.curImgFtr.throwDistance / self.projectedY

    # Augmentation Special Effects - default full sweep takes about two
    # seconds 52 frames - video is 26fps
    def wireframe(self, wireFrameProjection, frame, mainLane,
                  color=[255, 255, 255], wireThick=1,
                  sweepThick=5, fullsweepFrame=26):
        # calculate the wireframe positions
        nlanes = len(mainLane.lines) - 1
        leftPolynomial = np.poly1d(mainLane.lines[0].currentFit)
        roadleftPolynomial = np.poly1d(
            mainLane.lines[mainLane.left].currentFit)
        roadrightPolynomial = np.poly1d(
            mainLane.lines[mainLane.right].currentFit)
        rightPolynomial = np.poly1d(mainLane.lines[nlanes].currentFit)
        delta = (frame * 32) % 128
        squares = []

        # horizontal lines
        for i in range(int(self.projectedY / 128)):
            y1 = 128 * i + delta
            x1 = leftPolynomial([y1])
            x2 = roadleftPolynomial([y1])
            x3 = roadrightPolynomial([y1])
            x4 = rightPolynomial([y1])
            cv2.line(wireFrameProjection, (x1, y1),
                     (x4, y1), color, wireThick * 3)
            squares.append(((x2, y1), (x3, y1)))

        # vertical lines
        allY = [n * 32 for n in range(int(self.projectedY / 32))]
        polyDiff = np.polysub(mainLane.lines[nlanes].currentFit,
                              mainLane.lines[0].currentFit) / (nlanes * 2)
        curPoly = leftPolynomial
        for i in range(nlanes * 2):
            allX = curPoly(allY)
            XYPolyline = np.column_stack((allX, allY)).astype(np.int32)
            cv2.polylines(wireFrameProjection, [
                          XYPolyline], 0, color, int(wireThick / 4))
            curPoly = np.polyadd(curPoly, polyDiff)
        return squares

    # Augmentation Special Effects - default full sweep takes about four
    # seconds 104 frames - video is 26fps
    def sweep(self, wireFrameProjection, frame, lines,
              color=[0, 0, 255], sweepThick=5, fullsweepFrame=104):
        # calculate sweep angle
        halfcycle = fullsweepFrame / 2
        position = (frame % fullsweepFrame)
        if position > halfcycle:
            position = fullsweepFrame - position
        sweep = position / halfcycle
        allY = [n * 32 for n in range(int(self.projectedY / 32))]

        # calculate the wireframe positions
        nlanes = len(lines) - 1
        leftPolynomial = np.poly1d(lines[0].currentFit)
        rightPolynomial = np.poly1d(lines[nlanes].currentFit)

        # scanning sweep
        polySweepDiff = np.polysub(
            lines[nlanes].currentFit,
            lines[0].currentFit) * sweep
        sweepPoly = np.polyadd(leftPolynomial, polySweepDiff)
        allX = sweepPoly(allY)
        XYPolyline = np.column_stack((allX, allY)).astype(np.int32)
        cv2.polylines(wireFrameProjection, [XYPolyline], 0, color, sweepThick)
        sweepLane = 0
        for i in range(nlanes):
            leftLine = np.poly1d(lines[i].currentFit)
            rightLine = np.poly1d(lines[i+1].currentFit)
            if (leftLine([self.projectedY])[0] <
                    sweepPoly([self.projectedY])[0] and
                    sweepPoly([self.projectedY])[0] <
                    rightLine([self.projectedY])[0]):
                sweepLane = i
        return sweepLane

    def drawAxisOnLane(self, perspectiveImage):
        be_corner = np.float32(
            [[self.curDstRoadCorners[0][0],
              self.curDstRoadCorners[0][1], 0],
             [self.curDstRoadCorners[0][0],
              self.curDstRoadCorners[0][1], 0],
             [self.curDstRoadCorners[0][0],
              self.curDstRoadCorners[0][1], 0]]).reshape(-1, 3)

        be_axis = np.float32(
            [[self.curDstRoadCorners[0][0] + 64,
              self.curDstRoadCorners[0][1], 0],
             [self.curDstRoadCorners[0][0],
              self.curDstRoadCorners[0][1] + 128, 0],
             [self.curDstRoadCorners[0][0],
              self.curDstRoadCorners[0][1], -64]]).reshape(-1, 3)

        corner, jac = cv2.projectPoints(
            be_corner, self.rvecs, self.tvecs, self.mtx, self.dist)
        axis, jac = cv2.projectPoints(
            be_axis, self.rvecs, self.tvecs, self.mtx, self.dist)
        corner = tuple(corner[0].ravel())

        cv2.line(perspectiveImage, corner, tuple(
            axis[0].ravel()), (255, 0, 0), 5)
        cv2.line(perspectiveImage, corner, tuple(
            axis[1].ravel()), (0, 255, 0), 5)
        cv2.line(perspectiveImage, corner, tuple(
            axis[2].ravel()), (0, 0, 255), 5)

    def projectPoints(self, birdsEye3DPoints):
        m11 = self.dst2srcM[0][0]
        m12 = self.dst2srcM[0][1]
        m13 = self.dst2srcM[0][2]
        m21 = self.dst2srcM[1][0]
        m22 = self.dst2srcM[1][1]
        m23 = self.dst2srcM[1][2]
        m31 = self.dst2srcM[2][0]
        m32 = self.dst2srcM[2][1]
        m33 = self.dst2srcM[2][2]
        x = birdsEye3DPoints[:, 0]
        y = birdsEye3DPoints[:, 1]
        z = birdsEye3DPoints[:, 2]
        size = len(birdsEye3DPoints)
        perspectiveImagePoints = np.zeros((size, 2), dtype=np.float32)
        perspectiveImagePoints[:, 0] = (
            m11 * x + m12 * y + m13) / (m31 * x + m32 * y + m33)
        perspectiveImagePoints[:, 1] = (
            m21 * x + m22 * y + m23 - z) / (m31 * x + m32 * y + m33)
        return perspectiveImagePoints

    def drawCalibrationCube(self, perspectiveImage):
        be_cube = np.float32(
            [[self.curDstRoadCorners[0][0],
              self.curDstRoadCorners[0][1], 0],
             [self.curDstRoadCorners[1][0],
              self.curDstRoadCorners[1][1], 0],
             [self.curDstRoadCorners[2][0],
              self.curDstRoadCorners[2][1]-100, 0],
             [self.curDstRoadCorners[3][0],
              self.curDstRoadCorners[3][1]-100, 0],
             [self.curDstRoadCorners[0][0],
              self.curDstRoadCorners[0][1], self.z*2],
             [self.curDstRoadCorners[1][0],
              self.curDstRoadCorners[1][1], self.z*2],
             [self.curDstRoadCorners[2][0],
              self.curDstRoadCorners[2][1]-100, self.z*2],
             [self.curDstRoadCorners[3][0],
              self.curDstRoadCorners[3][1]-100, self.z*2]]).reshape(-1, 3)
        cube = self.projectPoints(be_cube)
        imgpts = np.int32(cube).reshape(-1, 2)

        # draw bottom of cube
        cv2.drawContours(perspectiveImage, [imgpts[:4]], -1, (255, 0, 0), 3)
        # draw sides of cube
        for i, j in zip(range(4), range(4, 8)):
            cv2.line(perspectiveImage, tuple(
                imgpts[i]), tuple(imgpts[j]), (0, 255, 0), 3)
        # draw top of cube
        cv2.drawContours(perspectiveImage, [imgpts[4:]], -1, (0, 0, 255), 3)

    def drawRoadSquares(self, perspectiveImage, squares):
        for square in squares:
            be_square = np.float32(
                [[square[0][0] - 16, square[0][1], 0],
                 [square[0][0] - 16, square[0][1], self.z*2],
                 [square[1][0] + 16, square[1][1], self.z*2],
                 [square[1][0] + 16, square[1][1], 0]]).reshape(-1, 3)
            square = self.projectPoints(be_square)
            imgpts = np.int32(square).reshape(-1, 2)
            # draw bottom of square
            cv2.drawContours(perspectiveImage, [
                             imgpts[:4]], -1, (0, 255, 0), 1)
