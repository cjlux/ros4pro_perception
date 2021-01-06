"""
This module contains the functions used for the preprocessing phase of the recognition.
"""
import os, glob, sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
from scipy.spatial import ConvexHull

import cv2

from skimage.transform import ProjectiveTransform, warp
from skimage.filters import threshold_otsu
from skimage.measure import approximate_polygon, find_contours, label, regionprops
from skimage.morphology import closing, convex_hull_object, square
from skimage.segmentation import clear_border

def show_image(im, title):
    """
    This function shows an image in grey scale.
    """
    plt.imshow(im, cmap='gray')
    plt.title(title)
    plt.show()

def show_binary(orig, binarized, title): 
    """
    This function shows the binarized version of the image
    """
    fig, ax = plt.subplots(2) 
    ax[0].imshow(orig, cmap='gray')
    ax[0].set_title("Original")
    ax[1].imshow(binarized, cmap='gray')
    ax[1].set_title("Binary")
    fig.suptitle(title)
    plt.show()

def binarize(image, debug=False):
    """
    This function returns a binarized version of the image.
    """
    # We make sure that we work on a local copy of the image
    img = image.copy()

    thresh = threshold_otsu(img)
    img = img > thresh
    img = closing(img, square(3))
    # remove artifacts connected to image border
    img = clear_border(img)
    # take only the frame of the cubes, not the inside
    img = convex_hull_object(img)

    if debug: show_image(img, "Binary image")

    return img

def get_box_contours(image, debug=False, area=300):
    """This fucntion finds the contours of the sques conatined in the image.
       Les tlis of contours is reordrered to follow the order left to right of
       the cubes in the image.
    """
    # We make sure that we work on a local copy of the image
    img = image.copy()

    img = binarize(img)

    # label image regions
    label_image = label(img)

    if debug:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image, cmap='gray')

    contours = []
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= area:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            contours.append([[maxc,maxr],[minc,maxr],[minc,minr],[maxc,minr]])
            if debug:
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
    
    if debug: plt.show()

    # re_order square contours to retrieve the order letf-to-right in teh image:
    contours = np.array(contours)
    order = np.argsort([ctr[1][0] for ctr in contours])
    ctrs = []
    for ctr in contours[order]:
        ctrs.append(ctr)

    return ctrs

def get_sprites(image, ctrs, debug=False):
    """
    This function computes a projective transform from the source (mnist image) to
    the destination (contour) and extracts the warped sprite.
    """
    # We make sure that we work on a local copy of the image
    img = image.copy()

    # We loop through the sprites
    sprts = []

    for contour in ctrs:

        # We compute the projective transform
        source_points = np.array([[28, 28], [0, 28], [0, 0], [28, 0]])
        destination_points = np.array(contour)
        transform = ProjectiveTransform()
        transform.estimate(source_points, destination_points)

        # We transform the image
        warped = warp(img, transform, output_shape=(28, 28))

        if debug:
            _, axis = plt.subplots(nrows=2, figsize=(8, 3))
            axis[0].imshow(img, cmap='gray')
            axis[0].plot(destination_points[:, 0], destination_points[:, 1], '.r')
            axis[0].set_axis_off()
            axis[1].imshow(warped, cmap='gray')
            axis[0].set_axis_off()
            plt.tight_layout()
            plt.show()

        sprts.append(warped)

    return sprts

def preprocess_sprites(sprts, debug=False):
    """
    This function preprocesses sprites to make them closer to the MNIST images.
    """

    out_sprites = []

    for n, img in enumerate(sprts):

        # We make a local copy
        img = img.copy()

        ##################
        # YOUR CODE HERE #
        ##################


 
        if debug: show_image(img, "Pre-processed sprites")           
        out_sprites.append(img)

    return out_sprites


if __name__ == "__main__":

    print("\n1) Loading images:")
    print("------------------")
    print(f"INFO: working directory is <{os.getcwd()}>")

    ###
    ### adjust directory "img_dir" as required:
    ###
    img_dir = '../data/ergo_cubes/'

    test_data = glob.glob(os.path.join(img_dir, '*.png'))
    print("Found test images: {}".format(test_data))
    images = []
    for path in test_data:
        # read image file and convert to gray:
        img = cv2.imread(path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        images.append(img)
    images = np.array(images)
    print(images[0].min(),images[0].max())
    show_image(images[0], "Image sample")
    input("Answer questions in 3.1 and press enter to continue... ")

    print("\n2) Binarizing image")
    print("-------------------")
    for im in images:
        binarize(im, debug=1)
    input("Answer questions in 3.2 and press enter to continue...")

    print("3) Getting boxes")
    print("----------------")
    ctrs = []
    for im in images:
        ctrs.append(get_box_contours(im, debug=1))
    input("Answer questions in 3.3 and press enter to continue...")

    print("4) Getting sprites")
    print("------------------")
    sprites = []
    for i in range(images.shape[0]):
        sprites.append(get_sprites(images[i], ctrs[i], debug=1))
    input("Answer questions in 3.4 and press enter to continue...")

    print("5) Pre-processing")
    print("-----------------")
    all_sprites = []
    for sprt in sprites:
        all_sprites.append(preprocess_sprites(sprt, debug=1))
    input("Answer questions in 3.5 and press enter to continue...")

