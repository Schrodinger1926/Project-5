import sys
from skimage.feature import hog
import cv2

image = cv2.imread(sys.argv[1])
print(image.shape)

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=True,
                     feature_vec=True):

    """
    cv2.imshow('before', img)
    cv2.waitKey(0)
    img = img*255
    cv2.imshow('after', img)
    cv2.waitKey(0)
    """
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
        return hog_features


hog_features = get_hog_features(image,
                                orient = 9,
                                pix_per_cell = 8,
                                cell_per_block = 2,
                                vis = False,
                                feature_vec = False)

print(hog_features.ravel().shape)
