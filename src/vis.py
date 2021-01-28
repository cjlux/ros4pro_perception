"""
Visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from seaborn import heatmap
from itertools import *
from tensorflow.keras import backend

def preview_samples(images, labels, title):
    """
    Shows a few samples of the loader.
    """

    # We generate a plot
    (fig, ax) = plt.subplots(7, 7, figsize=(5, 5))
    plt.subplots_adjust(hspace=0.5)
    for i, (x,y) in enumerate(product(range(7), range(7))):
        current_axis = ax[x, y]
        cm = current_axis.imshow(images[i].reshape(28, 28), cmap='gray')
        current_axis.set_title(labels[i].item())
        current_axis.set_axis_off()
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def preview_kernels(kernels, title):
    """
    Shows a few kernels in 2d
    """

    # We reshape kernels
    (w, h, a, b) = kernels.shape

    # We generate a plot
    (fig, ax) = plt.subplots(a, b)
    for x, y in product(range(a), range(b)):
        if a == 1:
            current_axis = ax[y]
        else:
            current_axis = ax[x, y]
        current_axis.imshow(kernels[:,:,x,y], cmap='gray')
        current_axis.set_axis_off()
    fig.suptitle(title)
    plt.show()

def plot_loss_accuracy(history):
    '''Plot training & validation loss & accuracy values, giving an argument
       'history' of type 'tensorflow.python.keras.callbacks.History'. '''
    
    plt.figure(figsize=(15,5))
    ax1 = plt.subplot(1,2,1)
    if history.history.get('accuracy'):
        ax1.plot(np.array(history.epoch)+1, history.history['accuracy'], 'o-',label='Train')
    if history.history.get('val_accuracy'):
        ax1.plot(np.array(history.epoch)+1, history.history['val_accuracy'], 'o-', label='Test')
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch') 
    ax1.grid()
    ax1.legend(loc='best')
    
    # Plot training & validation loss values
    ax2 = plt.subplot(1,2,2)
    if history.history.get('loss'):
        ax2.plot(np.array(history.epoch)+1, history.history['loss'], 'o-', label='Train')
    if history.history.get('val_loss'):
        ax2.plot(np.array(history.epoch)+1, history.history['val_loss'], 'o-',  label='Test')
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='best')
    ax2.grid()
    plt.show()


def plot_learning_curves(train_loss, test_loss, title):
    """
    Plot learning curves
    """
    fig, ax = plt.subplots()
    ax.plot(train_loss, color="blue", label="Train-set")
    ax.plot(test_loss, color="green", label="Test-set")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.set_title(title)
    ax.legend()
    plt.show()


def preview_activations(model, sample, title):
    """
    This function allows to preview the activations through the model.
    """

    activations = backend.function([model.layers[0].input],
                                   [model.layers[0].output,
                                    model.layers[1].output,
                                    model.layers[2].output,
                                    model.layers[3].output,
                                    model.layers[4].output,
                                    model.layers[5].output,
                                    model.layers[6].output,
                                    model.layers[7].output])
    output = activations(sample)

    fig = plt.figure()
    fig.suptitle(title)
    plt.subplots_adjust(hspace=0.5)
    gs = gridspec.GridSpec(8, 1)

    def add_volume_activation(n, title):
        act = output[n]
        w = act.shape[-1]
        g = gridspec.GridSpecFromSubplotSpec(1, w, subplot_spec=gs[n])
        for i in range(w):
            ax = fig.add_subplot(g[i])
            ax.imshow(act[0,:,:,i], cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            if float(w)/float(i+1) == 2:
                ax.set_title(title)

    add_volume_activation(0, "First Convolutional layer output")
    add_volume_activation(1, "First MaxPooling layer output")
    add_volume_activation(2, "Second Convolutional layer output")
    add_volume_activation(3, "Second MaxPooling layer output")

    def add_flat_activation(n, title):
        act = output[n]
        w = act.shape[-1]
        g = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[n])
        ax = fig.add_subplot(g[0])
        ax.imshow(act, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

    add_flat_activation(4, "Flatten layer output")
    add_flat_activation(5, "First Dense layer output")
    add_flat_activation(6, "Second Dense layer output")
    add_flat_activation(7, "Output layer output")

    plt.show()
