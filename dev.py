import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys
import glob
import pickle

from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

CAR_IMG_DIR = 'vehicles'
NOT_CAR_IMG_DIR = 'non-vehicles'

def find_total_files(folder):
    count = 0
    for root, subfolder, files in os.walk(folder):
        count += len(files)
    return count



def data_look():
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = find_total_files(CAR_IMG_DIR)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = find_total_files(NOT_CAR_IMG_DIR)
    # Read in a test image, either car or notcar
    # Define a key "image_shape" and store the test image shape 3-tuple
    #data_dict["image_shape"] = cv2.imread(car_list[0]).shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = np.uint8
    # Return data_dict
    return data_dict


def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

def color_hist(img, nbins = 32, bins_range = (0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=True,
                     feature_vec=True):

    #print("SHAPE: {}".format(img.shape))
    return_list = hog(img,
                      orientations = orient,
                      pixels_per_cell = (pix_per_cell, pix_per_cell),
                      cells_per_block = (cell_per_block, cell_per_block),
                      block_norm= 'L2-Hys', transform_sqrt=False,
                      visualize= vis, feature_vector= feature_vec)

    # name returns explicitly
    hog_features = return_list[0]

    if vis:
        hog_image = return_list[1]
        return hog_features, hog_image
    else:
        return hog_features.ravel()


def get_features(image, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
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

    # Apply color_hist() also with a color space option now
    hog_features = get_hog_features(feature_image,
                                    orient = 9,
                                    pix_per_cell = 8,
                                    cell_per_block = 2,
                                    vis = False,
                                    feature_vec = False)

    return np.concatenate((spatial_features, hist_features, hog_features))
    #return np.concatenate((spatial_features, hist_features))

def feature_extraction(data):
    X, y = [], []
    for label, folder in enumerate(data):
        for root, subfolder, files in os.walk(folder):
            for filename in files:
                if filename.endswith('.png'):
                    image_path = os.path.join(root, filename)
                    image = mpimg.imread(image_path)
                    X.append(get_features(image))
                    y.append(label)

    # Info
    X, y = np.array(X), np.array(y).reshape(-1, 1)
    # Shuffle data here
    print("shuffling")
    X, y = shuffle(X, y)

    print("Extraction Complete ..")
    print("Dataset: {} | Labels: {}".format(X.shape, y.shape))
    return X, y


def feature_scaling(X_train, X_test):
    print("scaling ..")
    X_scaler = StandardScaler().fit(X_train)

    # scale train
    scaled_X_train = X_scaler.transform(X_train)

    # scale test
    scaled_X_test = X_scaler.transform(X_test)
    print("done!")
    print("saving scaler")
    pickle.dump(X_scaler, open('scaler.sav', 'wb'))

    return scaled_X_train, scaled_X_test, X_scaler


def hyperparam_optimization():
    pass


def train(X_train, y_train):

    print("training ...")
    svc = LinearSVC(verbose = True)
    svc.fit(X_train, y_train)
    print("done!")
    print("saving model")
    pickle.dump(svc, open('model.sav', 'wb'))
    return svc


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = get_features(test_img)
        """
        features = get_features(test_img, cspace=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        """
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


def detect_boxes(image, svc, X_scaler):

    color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 0 # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [image.shape[0]//2, None] # Min and max in y to search in slide_window()

    draw_image = np.copy(image)
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    plt.imshow(window_img)
    plt.show()


def pipeline():
    print(data_look())
    X, y = feature_extraction(data = [CAR_IMG_DIR, NOT_CAR_IMG_DIR])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape)
    X_train, X_test, X_scaler = feature_scaling(X_train, X_test)
    svc = train(X_train, y_train)
    print("Accuracy: {}".format(svc.score(X_test, y_test)))
    return True


if __name__ == "__main__":
    #pipeline()
    print("Loading model params")
    svc = pickle.load(open('model.sav', 'rb'))
    X_scaler = pickle.load(open('scaler.sav', 'rb'))
    image = mpimg.imread(sys.argv[1])
    #image = image.astype(np.float32)/255
    #print(image)
    detect_boxes(image, svc, X_scaler)

