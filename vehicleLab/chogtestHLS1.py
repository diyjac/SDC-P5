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
from testlib.CHOG import CHOG

versionName = 'CHOGHLS1'
trained_model = './trained/'+versionName+'.pkl'
trained_scalar = './trained/scaler'+versionName+'.pkl'
visualfile = './visualized/'+versionName+'-augmented.jpg'

orient = 9
pix_per_cell = 8
cell_per_block = 2

# try to make thresholds LOWER!
# remove more false positives
# going down from 8.0
# going down from 7.0
# going down from 6.0
# going down from 5.0
# going up from 4.0
# going up from 4.5
# going up from 4.75
# locked
hthreshold = 4.85

# remove more false negatives
# going down from 1.5
# going down from 1.0
# going down from 0.9  <-- cannot see edge car, but can see car 3 and 4 in test 22
# going down from 0.8  <-- cannot see edge car, but can see car 3 and 4 in test 22
# going down from 0.7  <-- cannot see edge car, but can see car 3 and 4 in test 22
# going down from 0.6  <-- cannot see edge car, but can see car 3 and 4 in test 22
# going down from 0.5  <-- cannot see edge car, but can see car 3 and 4 in test 22
# going down from 0.4  <-- cannot see edge car, but can see car 3 and 4 in test 22
# going down from 0.0  <-- cannot see edge car, but can see car 3 and 4 in test 22
# going down from -1.0  <-- cannot see edge car, but can see car 3 and 4 in test 22
# going up from -5.0  <-- cannot see edge car, but can see car 2 3 4 (5) in test 22
lthreshold = -3.0

svc = joblib.load(trained_model) 
slide_window = SlidingWindows()
chog = CHOG(trained_scalar=trained_scalar)

outimages = []
images = glob.glob('./test_images/test*proj.jpg')
for file in images: 
    print("processing: ", file)
    image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)

    print("initializing...")
    windows = slide_window.completeScan(file)

    foundwindows = []
    print("Processing",len(windows),"windows high...")
    for window in windows:
        wimage = image[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        wfeatures = chog.extract_features(wimage, cspace='HLS', spatial_size=(32, 32),
                            orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                            hist_bins=32, hist_range=(0, 256), hog_channel=2)
        if wfeatures is not None:
            confidence = svc.decision_function(wfeatures.reshape(1, -1))
            if confidence[0] > hthreshold:
                foundwindows.append(window)
    window_img1 = chog.draw_boxes(image, foundwindows, color=(0, 0, 255), thick=2)

    foundwindows = []
    print("Processing",len(windows),"windows low...")
    for window in windows:
        wimage = image[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        wfeatures = chog.extract_features(wimage, cspace='HLS', spatial_size=(32, 32),
                            orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                            hist_bins=32, hist_range=(0, 256), hog_channel=2)
        if wfeatures is not None:
            confidence = svc.decision_function(wfeatures.reshape(1, -1))
            if confidence[0] > lthreshold:
                foundwindows.append(window)
    window_img2 = chog.draw_boxes(image, foundwindows, color=(0, 0, 255), thick=2)

    outimages.append((file, orient, pix_per_cell, cell_per_block, window_img1, window_img2))

chog.drawPlots(visualfile, versionName, outimages, hthreshold, lthreshold)

