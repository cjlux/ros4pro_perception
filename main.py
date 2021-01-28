"""
This script loads an image, extracts sprites, and infer the classes with the network trained earlier.
"""
import collections
import matplotlib.pyplot as plt
import glob
import cv2 

from src.detection import get_box_contours, get_sprites, preprocess_sprites
from tensorflow.keras.models import load_model

LABELS=[1, 2]

Box = collections.namedtuple('Box', 'contour sprite input label')

def process(image, model, debug=None):
    """
    This function processes an image given a model, and returns a list of Box.
    """

    debug_inner = debug in ["inner", "all"]
    contours = get_box_contours(image, debug=debug_inner)
    sprites = get_sprites(image, contours, debug=debug_inner)
    inputs = preprocess_sprites(sprites, debug=debug_inner)
    labels = [model.predict(i.reshape(1, 28, 28, 1)).squeeze().argmax() + 1 for i in inputs]

    boxes = [Box(contour=c, sprite=s, input=i, label=l) for c, s, i, l in  zip(contours, sprites, inputs, labels)]

    if debug in ["all", "synthesis"]:
        for box in boxes:
            fig, ax = plt.subplots(nrows=2)
            ax[0].imshow(image, cmap='gray')
            ax[0].plot(box.contour[:,0], box.contour[:,1], "og")
            ax[0].plot(box.contour.mean(axis=0)[0], box.contour.mean(axis=0)[1], "og")
            ax[0].set_axis_off()
            ax[1].imshow(box.sprite, cmap='gray')
            ax[1].imshow(box.input, cmap='gray')
            ax[1].set_title("Label recognized: {}".format(box.label))
            ax[1].set_axis_off()
            plt.tight_layout()
            
            plt.show()

    return boxes


if __name__ == "__main__":

    print("\n1) Load model")
    print("----------------")
    path = input("Enter the path to your network file: ")
    model = load_model(path)
    print(model.summary())

    print("\n2) Process data")
    print("------------")

    ###
    ### adjust directory "img_dir" as required:
    ###
    img_dir = './data/ergo_cubes/'

    test = glob.glob(img_dir + '*png')    
    for path in test:
        print("Testing image {}".format(path))
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            boxes = process(image, model, debug="synthesis")
            print('boxes:', boxes)
        except Exception as e:
            print(f"Failed to process image {<path>}")
            pass
