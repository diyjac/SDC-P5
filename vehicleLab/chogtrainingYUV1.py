import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
import glob
import time
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.externals import joblib
# NOTE: the next import is only valid for scikit-learn version >= 0.18
# for scikit-learn <= 0.17 use:
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

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
def extract_features(imgs, cspace='YUV', spatial_size=(32, 32),
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

# Define a way for us to write out a sample of the HOG
def drawPlots(imagefile, sampleTitle, orient, pix_per_cell, cell_per_block, trainScore, testScore, carimage, carhog, notcarimage, notcarhog, deltaTime):
    print("saving sample image and hogs to ", imagefile)
    # Setup plot
    fig = plt.figure(figsize=(10, 3))
    w_ratios = [1 for n in range(5)]
    h_ratios = [1 for n in range(1)]
    grid = gridspec.GridSpec(1, 5, wspace=0.0, hspace=0.0, width_ratios=w_ratios, height_ratios=h_ratios)
    i = 0

    # draw the images
    # next image
    sampleTitleWScores = '%s\n Orientation: %d\n Pix_per_cell: %d\n Cell_per_block: %d\n Train Accuracy:\n  %10.9f\n Test Accuracy:\n  %10.9f\n Decision Time:\n  %10.9f'%(sampleTitle, orient, pix_per_cell, cell_per_block, trainScore, testScore, deltaTime)
    ax = plt.Subplot(fig, grid[i])
    ax.text(0.1,0.4, sampleTitleWScores, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.add_subplot(ax)
    i += 1

    ax = plt.Subplot(fig, grid[i])
    ax.imshow(carimage)
    if i==1:
        ax.set_title('Sample Car Image', size=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    i += 1

    ax = plt.Subplot(fig, grid[i])
    ax.imshow(carhog, cmap='gray')
    if i==2:
        ax.set_title('Sample Car HOG', size=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    i += 1

    ax = plt.Subplot(fig, grid[i])
    ax.imshow(notcarimage)
    if i==3:
        ax.set_title('Sample Noncar Image', size=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    i += 1

    ax = plt.Subplot(fig, grid[i])
    ax.imshow(notcarhog, cmap='gray')
    if i==4:
        ax.set_title('Sample Noncar HOG', size=8)

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
orient = 9
pix_per_cell = 8
cell_per_block = 2

t=time.time()
car_features = extract_features(cars, cspace='YUV', spatial_size=(32, 32),
                        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hist_bins=32, hist_range=(0, 256), hog_channel=0, datatype='Car')
notcar_features = extract_features(notcars, cspace='YUV', spatial_size=(32, 32),
                        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hist_bins=32, hist_range=(0, 256), hog_channel=0, datatype='Noncar')
t2 = time.time()
print(t2-t, 'Seconds to load dataset...')

t=time.time()
print("Data loaded, now scaling and splitting dataset...")

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

t2 = time.time()
print(t2-t, 'Seconds to scale and split dataset...')

print("training set size:", len(X_train))
print("testing set size:", len(X_test))

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
versionName = 'CHOGYUV1'

# saving trained SVC model:
trained_model = './trained/'+versionName+'.pkl'
trained_scalar = './trained/scaler'+versionName+'.pkl'
visualfile = './visualized/'+versionName+'.jpg'

print('saving trained model to', trained_model) 
joblib.dump(svc, trained_model)
print('saving trained scalar to', trained_scalar)
joblib.dump(X_scaler, trained_scalar)

carimage, carhog = extract_features([cars[0]], cspace='YUV', spatial_size=(32, 32),
                        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hist_bins=32, hist_range=(0, 256), hog_channel=0, datatype='Car', visualize=True)
notcarimage, notcarhog = extract_features([notcars[0]], cspace='YUV', spatial_size=(32, 32),
                        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hist_bins=32, hist_range=(0, 256), hog_channel=0, datatype='Noncar', visualize=True)
drawPlots(visualfile, versionName, orient, pix_per_cell, cell_per_block, trainingScore, testingScore, carimage, carhog, notcarimage, notcarhog, deltatime)

