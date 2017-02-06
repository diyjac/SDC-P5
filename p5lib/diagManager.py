#!/usr/bin/python
"""
diagManager.py: version 0.1.0

History:
2017/01/29: coding style phase1:
            reformat to python-guide.org code style
            http://docs.python-guide.org/en/latest/writing/style/
            which uses PEP 8 as a base: http://pep8.org/.
2017/01/09: Initial version converted to a class
"""

import numpy as np
import cv2
import math
from p5lib.cameraCal import CameraCal
from p5lib.imageFilters import ImageFilters
from p5lib.projectionManager import ProjectionManager
from p5lib.line import Line


class DiagManager():
    # Initialize ImageFilter

    def __init__(self, roadManager):
        self.rMgr = roadManager
        self.pMgr = self.rMgr.projMgr

    ########################################################
    # Apply Textural Diagnostics
    ########################################################
    def textOverlay(self, diagScreen, offset, color=(64, 64, 0)):
        roadMgr = self.rMgr
        projMgr = self.pMgr
        imgFtr = self.rMgr.curImgFtr
        font = cv2.FONT_HERSHEY_COMPLEX

        # output image stats
        y = 30 + offset
        text = '%-28s%-28s%-28sBalance: %f' % (
            imgFtr.skyText, imgFtr.skyImageQ,
            imgFtr.roadImageQ, imgFtr.roadbalance)
        cv2.putText(diagScreen, text, (30, y), font, 1, color, 2)

        # output projection stats in the next two lines
        y += 30
        if imgFtr.horizonFound:
            text = 'Projection Last Top: %d'
            text += '   Road Horizon: %d'
            text += '   Vanishing Point: %d'
            text = text % (roadMgr.lastTop, imgFtr.roadhorizon,
                           projMgr.lane_info[7][1])
            cv2.putText(diagScreen, text, (30, y), font, 1, color, 2)
        else:
            text = 'Projection Last Top: %d   Horizon: NOT FOUND!'
            cv2.putText(diagScreen, text % (
                roadMgr.lastTop), (30, y), font, 1, color, 2)
        y += 30
        text = 'Road Backoff at: %d   Gap: %d   Visibility: %6.2fm'
        cv2.putText(diagScreen, text % (
            projMgr.curGradient,
            projMgr.curGradient - projMgr.gradient0,
            imgFtr.throwDistance), (30, y), font, 1, color, 2)

        # output restart stats
        y += 30
        text = 'Restart Count: %d   Projection Reset Count: %d' % (
            roadMgr.restartCount, roadMgr.resetProjectionCount)
        cv2.putText(diagScreen, text, (30, y), font, 1, color, 2)

        # output vehicle stats
        y += 30
        text = 'Vehicle Count: %d' % (
            len(roadMgr.vehicles))
        cv2.putText(diagScreen, text, (30, y), font, 1, color, 2)

        return diagScreen

    ########################################################
    # Full diagnostics of the RoadManager
    ########################################################
    def fullDiag(self, color=(128, 128, 0)):
        roadMgr = self.rMgr
        projMgr = self.pMgr
        imgFtr = self.rMgr.curImgFtr
        font = cv2.FONT_HERSHEY_COMPLEX

        diag2 = projMgr.diag2.astype(np.uint8)
        imgFtr.drawHorizon(diag2)
        middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)

        # curvature output
        if roadMgr.roadStraight:
            text = 'Estimated lane curvature: road nearly straight'
            cv2.putText(middlepanel, text,
                        (30, 60), font, 1, color, 2)
        elif roadMgr.radiusOfCurvature > 0.0:
            text = 'Estimated lane curvature: center is %fm to the right'
            cv2.putText(middlepanel, text % (
                roadMgr.radiusOfCurvature), (30, 60), font, 1, color, 2)
        else:
            text = 'Estimated lane curvature: center is %fm to the left'
            cv2.putText(middlepanel, text % (-roadMgr.radiusOfCurvature),
                        (30, 60), font, 1, color, 2)

        # center of road output
        if roadMgr.lineBasePos < 0.0:
            text = 'Estimated left of center: %5.2fcm'
            cv2.putText(middlepanel, text % (-roadMgr.lineBasePos * 1000),
                        (30, 90), font, 1, color, 2)
        elif roadMgr.lineBasePos > 0.0:
            text = 'Estimated right of center: %5.2fcm'
            cv2.putText(middlepanel, text % (roadMgr.lineBasePos * 1000),
                        (30, 90), font, 1, color, 2)
        else:
            cv2.putText(middlepanel, 'Estimated at center of road',
                        (30, 90), font, 1, color, 2)

        # assemble the screen
        diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        if roadMgr.diag1.shape[0] == 720:
            diagScreen[0:720, 0:1280] = roadMgr.diag1
        else:
            diagScreen[0:720, 0:1280] = cv2.resize(
                np.rot90(roadMgr.diag1), (1280, 720),
                interpolation=cv2.INTER_AREA)

        # image filters
        diagScreen[0:240, 1280:1600] = cv2.resize(
            imgFtr.diag1, (320, 240), interpolation=cv2.INTER_AREA)
        diagScreen[0:240, 1600:1920] = cv2.resize(
            imgFtr.diag2, (320, 240), interpolation=cv2.INTER_AREA)
        diagScreen[240:480, 1280:1600] = cv2.resize(
            imgFtr.diag3, (320, 240), interpolation=cv2.INTER_AREA)
        diagScreen[240:480, 1600:1920] = cv2.resize(
            imgFtr.diag4, (320, 240), interpolation=cv2.INTER_AREA) * 4

        diagScreen[600:1080, 1280:1920] = cv2.resize(
            roadMgr.final, (640, 480), interpolation=cv2.INTER_AREA)

        diagScreen[720:840, 0:1280] = middlepanel

        # projection
        diagScreen[840:1080, 0:320] = cv2.resize(
            diag2, (320, 240), interpolation=cv2.INTER_AREA)
        diagScreen[840:1080, 320:640] = cv2.resize(
            projMgr.diag1, (320, 240), interpolation=cv2.INTER_AREA)
        diagScreen[840:1080, 640:960] = cv2.resize(
            np.rot90(projMgr.diag3), (320, 240),
            interpolation=cv2.INTER_AREA)
        diagScreen[840:1080, 960:1280] = cv2.resize(
            np.rot90(projMgr.diag4), (320, 240),
            interpolation=cv2.INTER_AREA)

        return diagScreen

    ########################################################
    # Diagnostics of the Projection Manager (Single)
    ########################################################
    def projectionHD(self):
        projMgr = self.pMgr
        imgFtr = self.rMgr.curImgFtr

        # assemble the screen
        diagScreen = projMgr.diag3.astype(np.uint8)

        return diagScreen

    ########################################################
    # Diagnostics of the Projection Manager
    ########################################################
    def projectionDiag(self):
        projMgr = self.pMgr
        imgFtr = self.rMgr.curImgFtr

        diag2 = projMgr.diag2.astype(np.uint8)
        imgFtr.drawHorizon(diag2)

        # assemble the screen
        diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        diagScreen[0:540, 0:960] = cv2.resize(
            diag2, (960, 540), interpolation=cv2.INTER_AREA)
        diagScreen[0:540, 960:1920] = cv2.resize(projMgr.diag1.astype(
            np.uint8), (960, 540), interpolation=cv2.INTER_AREA)
        diagScreen[540:1080, 0:960] = cv2.resize(np.rot90(
            projMgr.diag3.astype(np.uint8)), (960, 540),
            interpolation=cv2.INTER_AREA)
        diagScreen[540:1080, 960:1920] = cv2.resize(np.rot90(
            projMgr.diag4.astype(np.uint8)), (960, 540),
            interpolation=cv2.INTER_AREA)

        return diagScreen

    ########################################################
    # Diagnostics of the Image Filters
    ########################################################
    def filterDiag(self):
        imgFtr = self.rMgr.curImgFtr

        # assemble the screen
        diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        diagScreen[0:540, 0:960] = cv2.resize(
            imgFtr.diag1, (960, 540), interpolation=cv2.INTER_AREA)
        diagScreen[0:540, 960:1920] = cv2.resize(imgFtr.diag2.astype(
            np.uint8), (960, 540), interpolation=cv2.INTER_AREA)
        diagScreen[540:1080, 0:960] = cv2.resize(imgFtr.diag3.astype(
            np.uint8), (960, 540), interpolation=cv2.INTER_AREA)
        diagScreen[540:1080, 960:1920] = cv2.resize(imgFtr.diag4.astype(
            np.uint8), (960, 540), interpolation=cv2.INTER_AREA)

        return diagScreen
