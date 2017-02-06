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

versionName = 'CHOG-TEST-XO'
visualfile = './visualized/'+versionName+'-augmented.jpg'

slide_window = SlidingWindows()
chog = CHOG()

outimages = []
images = glob.glob('./test_images/test*proj.jpg')
for file in images: 
    print("processing: ", file)
    image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)

    print("initializing...")
    windows = slide_window.completeScan(file)
    window_img = chog.draw_boxes(image, windows, color=(0, 0, 255), thick=2)
    outimages.append((file, 0, 0, 0, image, window_img))
chog.drawXOPlots(visualfile, versionName, outimages)

