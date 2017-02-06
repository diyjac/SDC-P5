import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
import glob
import time
from numpy import random
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.externals import joblib
# NOTE: the next import is only valid for scikit-learn version >= 0.18
# for scikit-learn <= 0.17 use:
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

# function to shift images x or y offsets
def shiftxy(image, xoffset, yoffset):
    rows,cols, depth = image.shape
    M = np.float32([[1,0,xoffset],[0,1,yoffset]])
    res = cv2.warpAffine(np.copy(image),M,(cols,rows))
    assert (res.shape[0] == cols)
    assert (res.shape[1] == rows)
    return res

# function to rotate images by given degrees
def rotate(image, degree):
    rows, cols, depth = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), degree, 1)
    res = cv2.warpAffine(image, M, (cols,rows))
    assert (res.shape[0] == rows)
    assert (res.shape[1] == cols)
    return res

# function to resize the image
def scale(image, ratio):
    rows, cols, depth = image.shape
    newrows = int(ratio*rows)
    newcols = int(ratio*cols)
    res = cv2.resize(image, (newrows, newcols), interpolation=cv2.INTER_AREA)
    if newrows*newcols > (rows*cols):
        # image is larger than rows*cols, randomly crop the image back to rows*cols
        xoffset = (newcols-cols)-int(random.random()*float(newcols-cols))
        yoffset = (newrows-rows)-int(random.random()*float(newrows-rows))
        cropped = res[xoffset:xoffset+cols, yoffset:yoffset+rows]
        res = cropped
    else:
        # image is smaller than before, randomly insert it into a rows*cols canvas
        if newrows*newcols < (rows*cols):
            tmpimage = np.copy(image)*0
            xoffset = (cols-newcols)-int(random.random()*float(cols-newcols))
            yoffset = (rows-newrows)-int(random.random()*float(rows-newrows))
            tmpimage[xoffset:newrows+xoffset, yoffset:newcols+yoffset] = res
            res = tmpimage
    assert (res.shape[0] == rows)
    assert (res.shape[1] == cols)
    return res

def gaussian_blur(img, kernel_size):
    # Applies a Gaussian Noise kernel
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def intensity(image, factor):
    maxIntensity = 255.0 # depends on dtype of image data
    phi = 1
    theta = 1
    image0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**factor
    return np.array(image0, dtype=np.uint8)

def noJitter(simage):
    # just return the non-jittered image
    return simage

def jitterA(simage):
    # set up the random jitter
    x = int(random.random() * 6) - 3
    y = int(random.random() * 6) - 3
    degree = int(random.random()*30.0)-15
    ratio = random.random()*0.5 + 1.0
    brightness = (random.random()*1.5)+0.5

    image = intensity(scale(rotate(shiftxy(simage,x,y),degree),ratio), brightness)
    return image

def jitterB(simage):
    # set up the random jitter
    x = int(random.random() * 12) - 6
    y = int(random.random() * 6) - 3
    degree = int(random.random()*30.0)-15
    ratio = random.random()*0.5 + 1.0
    kernel_size = int(random.random()*3)*2+1

    image = scale(rotate(shiftxy(simage,x,y),degree),ratio)
    return image

def jitterC(simage):
    # set up the random jitter
    x = int(random.random() * 4) - 2
    y = int(random.random() * 4) - 2
    degree = int(random.random()*30.0)-15
    ratio = random.random()*0.5 + 1.0

    image = scale(rotate(shiftxy(simage,x,y),degree),ratio)
    return image

def jitterD(simage):
    # set up the random jitter
    x = int(random.random() * 4) - 2
    y = int(random.random() * 4) - 2
    degree = int(random.random()*30.0)-15
    ratio = random.random()*0.5 + 1.0
    kernel_size = int(random.random()*2)
    brightness = (random.random()*1.5)+0.5
    if kernel_size == 0:
        image = intensity(scale(rotate(shiftxy(simage,x,y),degree),ratio),brightness)
    else:
        kernel_size = 3
        image = gaussian_blur(intensity(scale(rotate(shiftxy(simage,x,y),degree),ratio), brightness), kernel_size)
    return image

def jitterE(simage):
    # set up the random jitter
    x = int(random.random() * 6) - 3
    y = int(random.random() * 12) - 6
    degree = int(random.random()*40.0)-20
    ratio = random.random()*0.5 + 0.85
    brightness = (random.random()*1.5)+0.5
    image = intensity(scale(rotate(shiftxy(simage,x,y),degree),ratio),brightness)
    return image

def nextimageAndLabel(imagefileset, labelset, jitter=[noJitter, jitterA, jitterB, jitterC, jitterD, jitterE]):
    # randomly select an image from the list
    # randomly select a jitter to apply
    # read in the image and apply the jitter and return the results
    simageIdx = int(random.random()*len(imagefileset))
    jitterRoutine = int(random.random()*len(jitter))
    image = cv2.cvtColor(cv2.imread(imagefileset[simageIdx]), cv2.COLOR_BGR2RGB)
    return jitter[jitterRoutine](image), labelset[simageIdx]

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
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
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0, datatype='', visualize=False):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    images_pbar = tqdm(range(len(imgs)), desc='Loading '+datatype+' Dataset', unit=' features')
    for i in images_pbar:
        file = imgs[i]
        # Read in each one by one
        image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
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
        else: feature_image = np.copy(image)
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Call get_hog_features() with vis=False, feature_vec=True
        if visualize:
            hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=visualize, feature_vec=True)
            # print("hog_image: ", hog_image.shape, type(hog_image[0][0]), np.min(hog_image), np.max(hog_image))
            # print("image: ", image.shape, type(image[0][0][0]), np.min(image), np.max(image))
            minhog = np.min(hog_image)
            hog_image = hog_image - minhog
            maxhog = np.max(hog_image)
            hog_image = ((hog_image/maxhog)*255).astype(np.uint8)
            return image, hog_image
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    # Return list of feature vectors
    return features


# Define a function to extract features from a generated jittered list of images
# Have this function call bin_spatial() and color_hist() with jittered samples
def extract_jittered_features(X_train_file, y_train, batch=10000, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0, datatype='', visualize=False):
    # Create a list to append feature vectors to
    features = []
    labels = []
    # Iterate through the list of images
    batches_pbar = tqdm(range(batch), desc='Generating '+datatype+' Jittered Dataset', unit=' features')
    for b in batches_pbar:
        # Read in each one by one
        image, label = nextimageAndLabel(X_train_file, y_train)
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
        else: feature_image = np.copy(image)      
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Call get_hog_features() with vis=False, feature_vec=True
        if visualize:
            hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=visualize, feature_vec=True)
            # print("hog_image: ", hog_image.shape, type(hog_image[0][0]), np.min(hog_image), np.max(hog_image))
            # print("image: ", image.shape, type(image[0][0][0]), np.min(image), np.max(image))
            minhog = np.min(hog_image)
            hog_image = hog_image - minhog
            maxhog = np.max(hog_image)
            hog_image = ((hog_image/maxhog)*255).astype(np.uint8)
            return image, hog_image
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
        labels.append(label)
    # Return list of feature vectors
    return features, labels

# Define a way for us to write out a sample of the HOG
def drawPlots(imagefile, sampleTitle, orient, pix_per_cell, cell_per_block, trainScore, testScore, carimage, carhog, notcarimage, notcarhog, carJimage, carJhog, notcarJimage, notcarJhog, deltaTime):
    print("saving sample image and hogs to ", imagefile)
    # Setup plot
    fig = plt.figure(figsize=(10, 4))
    w_ratios = [1 for n in range(5)]
    h_ratios = [1 for n in range(2)]
    grid = gridspec.GridSpec(2, 5, wspace=0.0, hspace=0.0, width_ratios=w_ratios, height_ratios=h_ratios)
    i = 0

    # draw the images
    # next image
    sampleTitleWScores = '%s\nJittered Samples\n Orientation: %d\n Pix_per_cell: %d\n Cell_per_block: %d\n Train Accuracy:\n  %10.9f'%(sampleTitle, orient, pix_per_cell, cell_per_block, trainScore)
    ax = plt.Subplot(fig, grid[i])
    ax.text(0.1,0.3, sampleTitleWScores, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.add_subplot(ax)
    i += 1

    ax = plt.Subplot(fig, grid[i])
    ax.imshow(carJimage)
    if i==1:
        ax.set_title('Sample Car Image', size=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    i += 1

    ax = plt.Subplot(fig, grid[i])
    ax.imshow(carJhog, cmap='gray')
    if i==2:
        ax.set_title('Sample Car HOG', size=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    i += 1

    ax = plt.Subplot(fig, grid[i])
    ax.imshow(notcarJimage)
    if i==3:
        ax.set_title('Sample Noncar Image', size=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    i += 1

    ax = plt.Subplot(fig, grid[i])
    ax.imshow(notcarJhog, cmap='gray')
    if i==4:
        ax.set_title('Sample Noncar HOG', size=8)

    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    i += 1


    # next Row
    # draw the images
    # next image
    sampleTitleWScores = '%s\nTest Samples\n Orientation: %d\n Pix_per_cell: %d\n Cell_per_block: %d\n  Test Accuracy:\n  %10.9f\n Decision Time:\n  %10.9f'%(sampleTitle, orient, pix_per_cell, cell_per_block, testScore, deltaTime)
    ax = plt.Subplot(fig, grid[i])
    ax.text(0.1,0.3, sampleTitleWScores, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.add_subplot(ax)
    i += 1

    ax = plt.Subplot(fig, grid[i])
    ax.imshow(carimage)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    i += 1

    ax = plt.Subplot(fig, grid[i])
    ax.imshow(carhog, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    i += 1

    ax = plt.Subplot(fig, grid[i])
    ax.imshow(notcarimage)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    i += 1

    ax = plt.Subplot(fig, grid[i])
    ax.imshow(notcarhog, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    i += 1

    plt.savefig(imagefile)

# Divide up into cars and notcars
# NOTE: Using our own collected data from 'birds-eye' view
cars = glob.glob('../vehicles/*/*/*.jpg')
notcars = glob.glob('../non-vehicles/*/*/*.jpg')

print("number of original car samples: ", len(cars))
print("number of original non-car samples: ", len(notcars))
orient = 8
pix_per_cell = 4
cell_per_block = 2

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)

X = cars + notcars
y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))

X_train_filename, X_test_filename, y_original_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

t=time.time()
X_test = extract_features(X_test_filename, cspace='RGB', spatial_size=(32, 32),
                        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hist_bins=32, hist_range=(0, 256), datatype='Testing')
t2 = time.time()
print(t2-t, 'Seconds to load testing dataset...')

t=time.time()
X_train, y_train = extract_jittered_features(X_train_filename, y_original_train, batch=50000,cspace='RGB', spatial_size=(32, 32),
                        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hist_bins=32, hist_range=(0, 256), datatype='Training')
t2 = time.time()
print(t2-t, 'Seconds to generate jittered training dataset...')

t=time.time()
print("Jittered Data generated, now scaling dataset...")

print("training set size:", len(X_train))
print("testing set size:", len(X_test))

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)

# Apply the scaler to X_train
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

t2 = time.time()
print(t2-t, 'Seconds to scale dataset...')

print("start training SVC classifier on HOG features...")
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(t2-t, 'Seconds to train SVC...')
# Check the score of the SVC
trainingScore = svc.score(X_train, y_train)
testingScore = svc.score(X_test, y_test)
print('Train Accuracy of SVC = ', trainingScore)
print('Test Accuracy of SVC = ', testingScore)
# Check the prediction time for a single sample
t=time.time()
confidence = svc.decision_function(X_test[0].reshape(1, -1))
t2 = time.time()
deltatime = t2-t
print(deltatime, 'Seconds to run decision_function with SVC')

# versionName for this version
versionName = 'CHOGRGB4'

# saving trained SVC model:
trained_model = './trained/'+versionName+'.pkl'
trained_scalar = './trained/scaler'+versionName+'.pkl'
visualfile = './visualized/'+versionName+'.jpg'

print('saving trained model to', trained_model) 
joblib.dump(svc, trained_model)
print('saving trained scalar to', trained_scalar)
joblib.dump(X_scaler, trained_scalar)

carimage, carhog = extract_features([cars[0]], cspace='RGB', spatial_size=(32, 32),
                        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hist_bins=32, hist_range=(0, 256), datatype='Car', visualize=True)
notcarimage, notcarhog = extract_features([notcars[0]], cspace='RGB', spatial_size=(32, 32),
                        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hist_bins=32, hist_range=(0, 256), datatype='Noncar', visualize=True)
carjitteredimage, carjitteredhog = extract_jittered_features([cars[0]], np.ones(1), batch=1, cspace='RGB', spatial_size=(32, 32),
                        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hist_bins=32, hist_range=(0, 256), datatype='Car', visualize=True)
notcarjitteredimage, notcarjitteredhog = extract_jittered_features([notcars[0]], np.zeros(1), cspace='RGB', spatial_size=(32, 32),
                        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hist_bins=32, batch=1, hist_range=(0, 256), datatype='Noncar', visualize=True)
drawPlots(visualfile, versionName, orient, pix_per_cell, cell_per_block, trainingScore, testingScore, carimage, carhog, notcarimage, notcarhog, carjitteredimage, carjitteredhog, notcarjitteredimage, notcarjitteredhog, deltatime)

