"""
A package containing utility functions for computational MRI exercises.

Author          : Jinho Kim
Email           : jinho.kim@fau.de
"""

from pathlib import Path
from typing import Optional

import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from PIL import Image, ImageDraw, ImageFont


def _get_root():
    return Path(__file__).parent


def load_data(fname):
    root = _get_root()
    assert fname.endswith(".mat"), "The filename must contain the .mat extension"
    mat = scipy.io.loadmat(root / fname)
    return mat


def imshow(
    imgs: list,
    gt: Optional[np.ndarray] = None,
    titles: Optional[list] = None,
    suptitle: Optional[str] = None,
    root: Optional[Path] = None,
    filename: Optional[str] = None,
    fig_size: Optional[tuple] = None,
    save_indiv: bool = False,
    num_rows: int = 1,
    pos: Optional[list] = None,
    norm: Optional[float] = None,
    is_mag: bool = True,
    font_size: int = 15,
    font_color="yellow",
    font_weight="normal",
):
    """
    This function displays multiple images in a single row.
    Args:
        imgs:                       list of images to display
        gt:                         ground truth image, optional
        titles:                     list of titles for each image, optional
        suptitle:                   main title for the figure, optional
        root:                       Root path to save, optional
        filename:                   name of the file to save the figure, optional
        fig_size:                   figure size, default is (15,10)
        save_indiv:                 Save individual images or not
        num_rows:                   The number of rows of layout (a single row by default)
        pos:                        Position of images.
                                    ex) for 2x3 layout, [1,1,1,0,1,1] plots images like
                                                        ====================
                                                        img1    img2    img3
                                                                img4    img5
                                                        ====================
                                    ex) for 2x3 layout with gt given, [1,1,1,0,1,1] plots images like
                                                        ========================
                                                        gt  img1    img2    img3
                                                                    img4    img5
                                                        ========================
        norm:                       normalization factor, default is 1.0
        is_mag:                     plot images in magnitude scale or not (optional, default=True)
        font_size:                  font size for metric display, default is 20
        font_color:                 font color for metric display, default is yellow
                                    Available options are ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
        font_weight:                font weight for metric display, default is normal.
                                    Available options are ['normal', 'bold', 'heavy', 'light', 'ultralight', 'medium', 'semibold', 'demibold']
    """

    pos, num_cols = _get_pos(pos, num_rows=num_rows, num_imgs=len(imgs))

    # if gt is given, add 0 to the pos list
    if gt is not None:
        pos_tmp = []
        for i in range(num_rows):
            pos_tmp += [1 if i == 0 else 0] + pos[(len(pos) // num_rows) * i : (len(pos) // num_rows) * (i + 1)]
        num_cols = num_cols + 1
        pos = pos_tmp.copy()

    if fig_size is None:
        fig_size = (num_cols * 5, num_rows * 4 + 0.5)

    f = plt.figure(figsize=fig_size)
    titles = [None] * len(imgs) if titles is None else titles
    titles = ["Ground truth"] + titles if gt is not None else titles

    imgs = [gt] + imgs if gt is not None else imgs
    imgs = [np.abs(i) for i in imgs] if is_mag else imgs

    img_idx = 0
    for i, pos_indiv in enumerate(pos, start=1):
        ax = f.add_subplot(num_rows, num_cols, i)

        if pos_indiv == 0:
            img = np.ones_like(imgs[0], dtype=float)
            title = ""
        else:
            img = imgs[img_idx]
            title = titles[img_idx]
            img_idx += 1

        if gt is not None and i == 1:
            annotate_gt(ax, font_size, font_color, font_weight)

        if gt is not None and i > 1 and pos_indiv:
            annotate_metrics(imgs[0], img, ax, font_size, font_color, font_weight)

        if norm is None:
            ax.imshow(img, cmap="gray")
        else:
            norm_method = clr.Normalize() if norm == 1.0 else clr.PowerNorm(gamma=norm)
            ax.imshow(img, cmap="gray", norm=norm_method)
        ax.axis("off")
        ax.set_title(title)

    f.suptitle(suptitle) if suptitle is not None else f.suptitle("")

    if root is None:
        root = _get_root()
    if isinstance(root, str):
        root = Path(root)
    root = root / "Results"
    if not root.exists() and filename:
        root.mkdir(parents=True, exist_ok=True)

    if filename is None:
        plt.show()

    elif filename is not None:
        filename = Path(filename).stem
        print(f"Saving figure to {root}")
        plt.savefig(root / filename, bbox_inches="tight", pad_inches=0.3)
        plt.close(f)
        if save_indiv:
            for img, title in zip(imgs, titles):
                img = abs(img)
                title = title.split("\n")[0]
                plt.imshow(img, cmap="gray", norm=clr.PowerNorm(gamma=norm))
                plt.axis("off")
                plt.savefig(
                    root / f"{filename}_{title}",
                    bbox_inches="tight",
                    pad_inches=0.2,
                )


def _get_pos(pos, num_rows, num_imgs):
    num_cols = np.ceil(num_imgs / num_rows).astype(int)
    len_pos = num_rows * num_cols

    if pos is None:
        pos = [1] * num_imgs + [0] * (len_pos - num_imgs)
    else:  # if pos is given
        assert np.count_nonzero(pos) == num_imgs, "Givin pos are not matched to the number of given images"
        res = len_pos - len(pos)
        pos += [0] * res

    return pos, num_cols


def fft2c(x, axes=(-2, -1)):
    return fftshift(fft2(ifftshift(x, axes=axes), axes=axes, norm="ortho"), axes=axes)


def ifft2c(x, axes=(-2, -1)):
    return ifftshift(ifft2(fftshift(x, axes=axes), axes=axes, norm="ortho"), axes=axes)


def calc_nmse(trg, src):
    norm_factor = trg.max()
    trg = trg / norm_factor
    src = src / norm_factor

    # Calculate the squared differences between corresponding pixels
    squared_diff = np.sum((trg - src) ** 2)
    # Calculate the mean squared value of the first image
    mse_reference = np.sum(trg**2)
    # Calculate the NMSE
    nmse = squared_diff / mse_reference

    return nmse


def annotate_gt(ax, font_size, font_color, font_weight):
    text = f"RMSE"
    font_props = {"family": "monospace"}

    ax.annotate(
        text,
        xy=(1, 1),
        xytext=(-2, -2),
        fontsize=font_size,
        color=font_color,
        xycoords="axes fraction",
        textcoords="offset points",
        horizontalalignment="right",
        verticalalignment="top",
        fontweight=font_weight,
        fontproperties=font_props,
    )


def annotate_metrics(trg, src, ax, font_size, font_color, font_weight):
    rmse = calc_nmse(trg, src)

    text = f"{rmse*100:.2f}"

    ax.annotate(
        text,
        xy=(1, 1),
        xytext=(-2, -2),
        fontsize=font_size,
        color=font_color,
        xycoords="axes fraction",
        textcoords="offset points",
        horizontalalignment="right",
        verticalalignment="top",
        fontweight=font_weight,
    )


def plot_spocks(traj, nSpock, filename=None):
    """
    Plot the k-space trajectory of the spock sequence

    Args:
        traj:       k-space trajectory
        nSpock:     number of spokes
    """
    plt.plot(traj[:, :nSpock].real, traj[:, :nSpock].imag)
    plt.suptitle(f"k-space trajectory: {nSpock} spokes")
    plt.show()
    if filename is not None:
        root = _get_root() / "Results"
        plt.savefig(root / filename)
    plt.close()


def create_gif(recons: list, title: str, duration: int = 100):
    """
    Create a gif from the list of images.
    The gif image shows double the size of the reconstructed images.
    @param recons: [recon1, recon2, ..., reconN]
        Type        : list
        reconN      : ndarray (shape: rxc) containing float values
    @param title: Title of gif file
    @param duration: duration of gif
    """
    recon_gif = []
    for i, recon in enumerate(recons):
        recon = recon / recon.max() * 255  # Rescaling between 0-255
        tmp_img = Image.fromarray(recon).convert("L")
        tmp_img = tmp_img.resize(tuple(i * 2 for i in recon.shape))  # double the size
        font = ImageFont.truetype("DejaVuSans.ttf", size=30)  # Font style
        draw = ImageDraw.Draw(tmp_img)
        draw.text((0, 0), f"iter={i}", fill=255, font=font)
        recon_gif.append(tmp_img)

    save_dir = Path("Results")
    if not save_dir.exists():
        save_dir.mkdir(exist_ok=True, parents=True)
    save_dir = save_dir / title

    save_dir = save_dir.with_suffix(".gif") if save_dir.suffix != ".gif" else save_dir

    recon_gif[0].save(
        save_dir,
        format="GIF",
        append_images=recon_gif[1:],
        save_all=True,
        duration=duration,
        loop=0,
    )

def my_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.vdot(torch.flatten(a), torch.flatten(b)))
