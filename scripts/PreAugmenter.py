#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.util import random_noise
from tqdm import tqdm
from pprint import pformat
# from pprint import pprint
# from rich import print
# from rich.progress import Progress, SpinnerColumn, MofNCompleteColumn, TimeElapsedColumn
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from copy import deepcopy
# Setup logging
import logging


class CustomFormatter(logging.Formatter):
    grey = "\033[37m"
    cyan = "\033[96m"
    yellow = "\033[93m"
    red = "\033[31m"
    bold_red = "\033[91;1m"
    reset = "\033[0m"
    format = "%(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: cyan + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
logger.propagate = True

import tensorflow as tf
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def set_tf_loglevel(level):
    """Set tensorflow log level on the fly."""
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)


def cartesian_product(*arrays):
    """Excellent cartesian product function from here:
    https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def find_grid_indices(sampledim, imgdim, num=-1, focusmiddle=False):
    """Find the indices along a dimension of an image at which to take samples.

    Args:
        sampledim (int): Dimension length of the given side of a sample (height or width).
        imgdim (int): Dimension length of the given side of the image.
        num (int): Number of samples to *attempt* to take from the image. Usually fewer are taken.
        focusmiddle (bool): Focuses samples on the middle of the image if True.

    Returns:
        numpy.array: Indices along a dimension of the image to take slices from.
    """
    if num > (imgdim // sampledim):
        # Modify distvec step size to be sampledim/(num/(imgdim // sampledim))
        # In this case because s / (n / (i / s)) == i / n, we just take i / n as our step
        # and generate the numbers accordingly
        distvec = np.arange(0, imgdim - sampledim, imgdim // num).astype(int)
        totalremainder = imgdim - (distvec[-1] + sampledim)
        startpad = int(np.floor(totalremainder / 2))
        distvec += startpad
        return distvec

    distvec = np.arange(0, imgdim - sampledim, sampledim).astype(int)
    if 0 < num <= distvec.size:
        distvec = distvec[0:num]
    # Find remaining pixels
    totalremainder = imgdim - (distvec[-1] + sampledim)

    if focusmiddle:
        startpad = int(np.floor(totalremainder / 2))
        distvec += startpad
    else:
        # Split remainder into n+1 individual gaps (so there's one on the left and right)
        gap = totalremainder // (distvec.size + 1)

        # Shift everything a bit to pad the start
        startpad = int(np.floor((totalremainder % (distvec.size + 1)) / 2))
        distvec += startpad

        # Add gaps to all starting indices
        gapvec = np.arange(1, distvec.size + 1, 1) * gap
        distvec += gapvec

    return distvec


def plot_patch_stack(stack, patchdims, page_layout=(4, 6), pages=1):
    """Plot pages of images from a 4D stack of images.

    Args:
        stack (4D numpy.ndarray): A stack of images to plot.
        patchdims (tuple): The shape of the slices of the image.
        page_layout (tuple): The (row,col) representation of the layout of a single page.
        pages (int): Number of pages to plot.
    """
    perpage = page_layout[0] * page_layout[1]  # (just for now)
    maxplot = perpage * pages
    plotstack = stack[:maxplot, :, :, :]

    for p in range(pages):
        page = np.zeros((page_layout[0] * patchdims[0], page_layout[1] * patchdims[1], 3), dtype=np.int64)
        pagestack = plotstack[p * perpage:(p + 1) * perpage, :, :, :]
        colcount = 0
        rowcount = 0
        for patch in range(pagestack.shape[0]):
            page[rowcount * patchdims[0]:(rowcount + 1) * patchdims[0],
            colcount * patchdims[1]:(colcount + 1) * patchdims[1], :] = pagestack[patch, :, :, :]
            colcount += 1
            if colcount >= page_layout[1]:
                colcount = 0
                rowcount += 1
        # Create figure and axes
        fig, ax = plt.subplots()
        # Display the image
        ax.imshow(page)
        plt.show()


def split_image(h, w, image, numsuggestion=-1, numperdim=-1, focusmiddle=False, plotit=False):
    """Split image into component tiles

    Args:
        h (int): Height of the tile.
        w(int): Width of the tile.
        image (numpy.array): An image to load.
        numsuggestion (int): A number of tiles for the splitter to try and generate. It will usually generate slightly fewer than you ask for.
        numperdim (int): The number of tiles to generate over both dimensions if numsuggestion is not set.
        focusmiddle (bool): Focus tiles on the middle of the image (particularly useful if you are only taking a few tiles and want to make sure you get the focus).
        plotit (bool): Plot the image along with the tile locations.

    Returns:
        4D numpy.ndarray: numtiles x image_height x image_width x 3 array of slices stacked on top of each other (a 4D array).
    """
    # Find height and width of image
    imgheight, imgwidth, _ = image.shape
    if numsuggestion > 0:
        # Calculate a proportionally overlapping grid to reduce weird periods in the overlaps
        # Could instead do e.g. num_h = np.sqrt(numsuggestion * imgheight * imgwidth) / imgwidth
        # Though this one is probably easier to understand
        linearatio = np.sqrt(numsuggestion / (imgheight * imgwidth))
        num_w = int(np.floor(imgwidth * linearatio))
        num_h = int(np.floor(imgheight * linearatio))
        gridindices_w = find_grid_indices(w, imgwidth, num_w, focusmiddle)
        gridindices_h = find_grid_indices(h, imgheight, num_h, focusmiddle)
    else:
        # Default behaviour
        gridindices_w = find_grid_indices(w, imgwidth, numperdim, focusmiddle)
        gridindices_h = find_grid_indices(h, imgheight, numperdim, focusmiddle)

    gridindices = cartesian_product(gridindices_h, gridindices_w)

    #     print(gridindices)

    if plotit:
        # Create figure and axes
        fig, ax = plt.subplots()
        # Display the image
        ax.imshow(image)
        # Create Rectangle patches
        pcollect_raw = [patches.Rectangle(x[::-1], h, w) for x in gridindices]
        pcollect = PatchCollection(pcollect_raw, linewidth=1, edgecolor='r', facecolor='none')
        # Add the collection to the Axes
        ax.add_collection(pcollect)
        plt.show()

    # Split image
    # There's probably a better way of doing this but honestly, this works just fine and is really quick
    img_stack = np.array([image[i[0]:i[0] + h, i[1]:i[1] + w, :] for i in gridindices])
    return img_stack


def generate_rotations(img_stack):
    """Generate all 90 degree rotations of a stack of images about their xy plane.

    Args:
        img_stack (4D numpy.ndarray): A stack of images to rotate.

    Returns:
        4D numpy.ndarray: A larger stack of images incorporating both the original images and their rotations.
    """
    rot90 = np.rot90(deepcopy(img_stack), k=1, axes=(1, 2))
    rot180 = np.rot90(deepcopy(img_stack), k=2, axes=(1, 2))
    rot270 = np.rot90(deepcopy(img_stack), k=3, axes=(1, 2))

    return np.concatenate([img_stack, rot90, rot180, rot270], axis=0)


def flipit(img_stack):
    """Horizontally flip a stack of images around their x axis.

    Args:
        img_stack (4D numpy.ndarray): A stack of images to flip.

    Returns:
        4D numpy.ndarray: A larger stack of images incorporating both the original images and their flipped variants.
    """
    # Horizontal flip
    flipped = np.flip(deepcopy(img_stack), axis=2)
    return np.concatenate([img_stack, flipped], axis=0)


def add_noise(img_stack):
    """Add RGB S&P noise to a stack of images.

    Currently unused!

    Args:
        img_stack (4D numpy.ndarray): A stack of images to noise.

    Returns:
        4D numpy.ndarray: A larger stack of images incorporating both the original images and their noised variants.
    """
    # This is the slowest transform currently at ~1s
    noised = random_noise(deepcopy(img_stack), mode="s&p", amount=0.015)
    # noised = np.apply_over_axes(preconf_randnoise, deepcopy(img_stack), [0])
    return np.concatenate([img_stack, noised], axis=0)


def add_random_brightness_toall(img_stack, max_delta=0.025):
    """Randomly vary the brightness of a stack of images per-image.

    Args:
        img_stack (4D numpy.ndarray): A stack of images to modify.
        max_delta (float): The max brightness modification that can be applied (+/-).

    Returns:
        4D numpy.ndarray: The input stack of images with modified brightness.
    """
    tensor_rep = tf.convert_to_tensor(img_stack)
    tensor_list = tf.unstack(tensor_rep)
    for i, x in enumerate(tensor_list):
        tensor_list[i] = tf.image.random_brightness(x, max_delta=max_delta)
    tensor_rep = tf.stack(tensor_list)
    return tensor_rep.numpy()


def add_random_contrast_toall(img_stack, lower=0, upper=1.125):
    """Randomly vary the contrast of a stack of images per-image.

    Args:
        img_stack (4D numpy.ndarray): A stack of images to modify.
        lower: The lower bound of the contrast adjustment.
        upper: The upper bound of the contrast adjustment.

    Returns:
        4D numpy.ndarray: The input stack of images with modified contrast.
    """
    tensor_rep = tf.convert_to_tensor(img_stack)
    tensor_list = tf.unstack(tensor_rep)
    for i, x in enumerate(tensor_list):
        tensor_list[i] = tf.image.random_contrast(x, lower=lower, upper=upper)
    tensor_rep = tf.stack(tensor_list)
    return tensor_rep.numpy()


def save_images(path, imgname, img_stack, postfix="", compress_level=3):
    """Save images individually from a stack to a path.

    Args:
        path (str): The path to save files to.
        imgname (str): The base name of the originating image.
        img_stack (4D numpy.ndarray): A stack of images to save.
        postfix (str): Any text to add after the uid of the slice and before the extension (e.g. '_mask').
        compress_level (int): The level of png compression to apply when saving (0-9 inclusive).
    """
    for i in range(img_stack.shape[0]):
        imsave(f"{os.path.join(path, imgname)}_{i}{postfix}.png", img_stack[i, :, :, :],
               check_contrast=False,
               plugin='pil',  # Using PIL to enable compression level argument, and thus giving approx a 3x speedup
               compress_level=compress_level)


def main(idir, imaskdir, odir, omaskdir, h, w, numsuggestion=-1, numperdim=-1, focusmiddle=False,
         maxdelta=0.025, contrastlower=0, contrastupper=1.125, pngcompression=3):
    """Augment a folder of images by slicing into patches, then rotating, flipping, and modifying brightness and colour.

    Args:
        idir (str): Input image directory.
        imaskdir (str): Input mask directory.
        odir (str): Output image directory.
        omaskdir (str): Output mask directory.
        h (int): Height of a patch in px.
        w (int): Width of a patch in px.
        numsuggestion (int): A number of tiles for the splitter to try and generate. It will usually generate slightly fewer than you ask for.
        numperdim (int): The number of tiles to generate over both dimensions if numsuggestion is not set.
        focusmiddle (bool): Focus tiles on the middle of the image (particularly useful if you are only taking a few tiles and want to make sure you get the focus).
        maxdelta (float): The max brightness modification that can be applied (+/-)
        contrastlower (float): The lower bound of the contrast adjustment.
        contrastupper (float): The upper bound of the contrast adjustment.
        pngcompression (int): The level of png compression to apply when saving (0-9 inclusive).
    """
    # Check if outdirs exist and create them if needed
    logger.info("Checking dirs...")
    if not os.path.exists(odir):
        logger.warning("Output dir does not exist! Creating...")
        os.makedirs(odir)
    if not os.path.exists(omaskdir):
        logger.warning("Output mask dir does not exist! Creating...")
        os.makedirs(omaskdir)
    logger.debug("Dir check complete.")

    # Construct dicts of images
    logger.info("Finding images...")
    images = {os.path.splitext(os.path.basename(file))[0]: file for file in os.listdir(idir) if
              file.lower().endswith((".jpg", ".png"))}
    masks = {os.path.splitext(os.path.basename(file))[0].rstrip("_mask"): file for file in os.listdir(imaskdir) if
             file.lower().endswith(".png")}
    logger.debug(f"Found {len(images)} images and {len(masks)} masks.")

    inboth = set(images).intersection(set(masks))

    # Check for differences between the two
    if len(inboth) < len(images):
        logger.warning("Images missing masks: %s", ', '.join(set(images).difference(set(masks))))
    if len(inboth) < len(masks):
        logger.warning("Masks missing images: %s", ', '.join(set(masks).difference(set(images))))

    # Construct img-mask pairs
    imgmaskpair_list = [[k, images[k], masks[k]] for k in inboth]

    total_augs = 0
    logger.info(f"Augmenting {len(imgmaskpair_list)} image-mask pairs")
    # Iterate over img-mask pairs
    with tqdm(total=len(imgmaskpair_list * 2), bar_format='{l_bar}{bar}{r_bar}') as t:
        for imagegroup in imgmaskpair_list:
            imgbase, imgname, maskname = imagegroup
            t.postfix = imgbase
            image = imread(os.path.join(idir, imgname))
            mask = imread(os.path.join(imaskdir, maskname))
            # Process image
            imagestack = split_image(h, w, image, numsuggestion, numperdim, focusmiddle, plotit=False)
            imagestack = flipit(imagestack)
            imagestack = generate_rotations(imagestack)
            imagestack = add_random_contrast_toall(imagestack, contrastlower, contrastupper)
            imagestack = add_random_brightness_toall(imagestack, maxdelta)
            save_images(odir, imgbase, imagestack, compress_level=pngcompression)
            total_augs += imagestack.shape[0]
            t.update()
            t.postfix = f"{imgbase}_mask"
            # Process mask
            maskstack = split_image(h, w, mask, numsuggestion, numperdim, focusmiddle, plotit=False)
            maskstack = flipit(maskstack)
            maskstack = generate_rotations(maskstack)
            save_images(omaskdir, imgbase, maskstack, postfix="_mask", compress_level=pngcompression)
            t.update()
        t.postfix = "Done!"

    # Alternative timing implementation in rich rather than tqdm
    # # Iterate over img-mask pairs
    # with Progress(
    #         SpinnerColumn("moon"),
    #         *Progress.get_default_columns(),
    #         MofNCompleteColumn(),
    #         TimeElapsedColumn(),
    #         auto_refresh=True
    #      ) as progress:
    #
    #     task = progress.add_task(f"[red]Augmenting {imgmaskpair_list[0][0]}...", total=len(imgmaskpair_list*2))
    #
    #     for imagegroup in imgmaskpair_list:
    #         imgbase, imgname, maskname = imagegroup
    #         progress.update(task, description=f"[red]Augmenting {imgname}...")
    #         progress.refresh()
    #         image = imread(os.path.join(idir, imgname))
    #         mask = imread(os.path.join(imaskdir, maskname))
    #         # Process image
    #         imagestack = split_image(h, w, image, numsuggestion, numperdim, focusmiddle, plotit=False)
    #         imagestack = flipit(imagestack)
    #         imagestack = generate_rotations(imagestack)
    #         imagestack = add_random_contrast_toall(imagestack, contrastlower, contrastupper)
    #         imagestack = add_random_brightness_toall(imagestack, maxdelta)
    #         save_images(odir, imgbase, imagestack, compress_level=pngcompression)
    #         total_augs += imagestack.shape[0]
    #         progress.update(task, advance=1)
    #         progress.update(task, description=f"[red]Augmenting {maskname}...")
    #         progress.refresh()
    #         # Process mask
    #         maskstack = split_image(h, w, mask, numsuggestion, numperdim, focusmiddle, plotit=False)
    #         maskstack = flipit(maskstack)
    #         maskstack = generate_rotations(maskstack)
    #         save_images(omaskdir, imgbase, maskstack, postfix="_mask", compress_level=pngcompression)
    #         progress.update(task, advance=1)
    #         progress.refresh()
    #     progress.update(task, description=f"[red]Done Augmenting!")
    #     progress.refresh()

    logger.info(f"Augmented {len(imgmaskpair_list)} pairs, creating a total test dataset of {total_augs} images")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process photos ready for u-net training.")
    parser.add_argument("imagedir", help="folder of images to process.")
    parser.add_argument("maskdir", help="folder of masks to process (in the form '<imgname>_mask.png').")
    parser.add_argument("outdir", help="folder to store augmented images in.")
    parser.add_argument("outmaskdir", help="folder to store augmented masks in.")
    parser.add_argument("width", help="width of slices to take from the images in px (int).", type=int)
    parser.add_argument("height", help="height of slices to take from the images in px (int).", type=int)

    parser.add_argument("-d", "--debug", action="store_true", help="enable debugging information")

    paramgroup = parser.add_argument_group("slicing parameters")
    paramgroup.add_argument("-n", "--numsuggest", nargs="?", type=int, const=-1, default=-1,
                            help="number of slices to take from each image. Note: you usually get a few fewer.",
                            metavar="int")
    paramgroup.add_argument("--numperdim", nargs="?", type=int, const=-1, default=-1,
                            help="number of slices to take from each dimension (superseded by numsuggest).",
                            metavar="int")
    paramgroup.add_argument("-m", "--focusmiddle", action="store_true", help="focus slices on the middle of the image.")

    auggroup = parser.add_argument_group("augmentation parameters")
    auggroup.add_argument("-b", "--maxdelta", nargs="?", type=float, const=0.025, default=0.025,
                          help="max brightness augmentation modification range (defaults to 0.025).",
                          metavar="flt")
    auggroup.add_argument("-l", "--contrastlower", nargs="?", type=float, const=0.0, default=0.0,
                          help="min contrast augmentation modification range (defaults to 0.0).",
                          metavar="flt")
    auggroup.add_argument("-u", "--contrastupper", nargs="?", type=float, const=1.125, default=1.125,
                          help="max contrast augmentation modification range (defaults to 1.125).",
                          metavar="flt")
    exgroup = parser.add_argument_group("export parameters")
    exgroup.add_argument("-p", "--pngcompression", nargs="?", type=int, const=3, default=3, choices=range(0, 10),
                         help="png compression level. Higher numbers give smaller pngs but slow down image export significantly.",
                         metavar="int")

    arglist = parser.parse_args()
    arglist = vars(arglist)

    if arglist["debug"]:
        logger.setLevel(logging.DEBUG)
        logger.warning("Running in debug mode...")
        logger.debug("Args: \n%s", pformat(arglist))

    logger.debug("Tensorflow logging is set to only log FATAL errors.")
    logger.debug("If you are having trouble, comment out the os.environ command at the top of PreAugmenter.py")

    set_tf_loglevel(logging.FATAL)
    try:
        main(arglist["imagedir"], arglist["maskdir"], arglist["outdir"], arglist["outmaskdir"],
             arglist["height"], arglist["width"], arglist["numsuggest"], arglist["numperdim"], arglist["focusmiddle"],
             arglist["maxdelta"], arglist["contrastlower"], arglist["contrastupper"], arglist["pngcompression"])
    except KeyboardInterrupt:
        logger.info("Exiting...")
    set_tf_loglevel(logging.INFO)
    sys.exit()
