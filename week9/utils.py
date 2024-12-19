"""
A package containing utility functions for computational MRI exercises.

Author          : Jinho Kim
Email           : jinho.kim@fau.de
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _get_root():
    return Path(__file__).parent

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def plot(
    vectors: list,
    labels: Optional[list] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    smoothing: Optional[int] = 1,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    root: Optional[Path] = None,
    filename: Optional[str] = None,
):
    """
    This function plots multiple vectors in a single figure.
    Args:
        vectors:            list of images to display
        labels:             list of labels for each image, optional
        xlabel:             label for x-axis, optional
        ylabel:             label for y-axis, optional
        title:              title for the figure, optional
        smoothing:          smoothing factor for the plot, optional
        xlim:               x-axis limits, optional
        ylim:               y-axis limits, optional
        root:               Root path to save, optional
        filename:           name of the file to save the figure, optional        
        watermark:          Add watermark or not
    """
    if not isinstance(vectors, list):
        vectors = [vectors]

    f, a = plt.subplots(1)
    for vector, label in zip(vectors, labels):
        a.plot(np.convolve(vector, np.ones((smoothing,)) / smoothing, mode='valid'), label=label)

    if xlabel:
        a.set_xlabel(xlabel)
    if ylabel:
        a.set_ylabel(ylabel)
    if title:
        a.set_title(title)
    if xlim:
        a.set_xlim(xlim)
    if ylim:
        a.set_ylim(ylim)
    a.legend()

    if root is None:
        root = _get_root()
    if isinstance(root, str):
        root = Path(root)
    root = root / "Results"
    if not root.exists() and filename:
        root.mkdir(parents=True, exist_ok=True)

    if filename is None:
        plt.show()
    else:
        plt.savefig(root/f"{filename}_{title}", bbox_inches="tight", pad_inches=0.2)
    plt.close()
