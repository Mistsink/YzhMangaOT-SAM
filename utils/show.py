from typing import Optional
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from cv2 import Mat
from torch import Tensor
from numpy import ndarray


def show_mask(mask: Tensor, ax: Axes, random_color=False):
    '''
    mask: (h, w)
    '''
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(points: ndarray, ax: Axes, labels=None, marker_size=375):
    '''
    points: (n, 2)
    '''
    if labels is None:
        labels = np.ones(points.shape[0])
    pos_points = points[labels==1]
    neg_points = points[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


def show_image(image: Mat, figsize=(10, 10),  points: Optional[ndarray]=None, mask: Optional[Tensor]=None, title=''):
    '''
    points: (n, 2)
    mask: (h, w)
    '''
    fig, axes = plt.subplots(figsize=figsize)

    axes.imshow(np.array(image))
    if points is not None:
        show_points(points, axes)
    if mask is not None:
        show_mask(mask, axes)
    axes.title.set_text(title)
    axes.axis("off")