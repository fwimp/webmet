import skimage as img

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage import io
from skimage import filters
from skimage.feature import corner_harris, corner_peaks, corner_shi_tomasi, corner_fast, corner_kitchen_rosenfeld
from skimage.exposure import rescale_intensity
from scipy.spatial import cKDTree
import os
# NJA is currently a private package, but should be available via git
import NJA
from skimage.color import rgb2gray

import logging
logger = logging.getLogger(__name__)
logger.propagate = True


def find_image_palette(image, clusters=5, max_iter=200, epsilon=.1, attempts=10):
    pixels = np.float32(image.reshape(-1, 3))

    logger.debug("k-means params: clusters={}, max_iter={}, epsilon={}, attempts={}.".format(clusters, max_iter, epsilon, attempts))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, clusters, None, criteria, attempts, flags)
    _, counts = np.unique(labels, return_counts=True)
    return palette, labels, counts


def estimate_contrasting_colour(palette, return_idx=False):
    lab_palette = img.color.rgb2lab(np.array([palette]))
    delta = np.asmatrix([[img.color.deltaE_cie76(x, y) for x in lab_palette[0]] for y in lab_palette[0]])
    delta_mean = np.asarray(np.mean(delta, axis=1).flatten())[0]
    idx = np.where(delta_mean == np.amax(delta_mean))[0][0]
    if return_idx:
        return idx
    return palette[idx]


def hex2rgb_single(colour):
    colour_raw = colour.lstrip("#")
    if len(colour_raw) == 6:
        colour_raw = np.asarray([int(colour_raw[i:i + 2], 16) for i in range(0, len(colour_raw), 2)], dtype=np.float32)
    elif len(colour_raw) == 3:
        colour_raw = np.asarray([int(colour_raw[i] * 2, 16) for i in range(0, len(colour_raw))], dtype=np.float32)
    else:
        raise ValueError('colour arg must be a valid hex RGB code. Provided "{}"'.format(colour))
    return colour_raw


def closest_contrasting_colour(palette, idcolour, hexcolour=False, showwarnings=True, return_idx=False):
    if hexcolour:
        idcolour = hex2rgb_single(idcolour)
    lab_palette = img.color.rgb2lab(np.array([palette]))
    try:
        lab_idcolour = img.color.rgb2lab(np.array([np.array([idcolour])]))
    except ValueError:
        if showwarnings:
            print("Invalid idcolour: {}\nAttempting with conversion from hexcode (did you forget to provide hexcolour=True?)".format(idcolour))
        idcolour = hex2rgb_single(idcolour)
        lab_idcolour = img.color.rgb2lab(np.array([np.array([idcolour])]))
    delta_mean = np.asarray([img.color.deltaE_cie76(x, lab_idcolour) for x in lab_palette[0]])
    idx = np.where(delta_mean == np.amin(delta_mean))[0][0]
    if return_idx:
        return idx
    return palette[idx]


def merge_processed_images(dusting, edgemap, a=0.5, b=0.5, maskthreshold=0.25):
    logger.debug("Weighted average params: a={}, b={}".format(a, b))
    if a + b > 1:
        raise ValueError("weightings a and b must not add up to > 1!")

    # Normalise inputs
    dusting = rescale_intensity(dusting, out_range=(0., 1.))
    edgemap = rescale_intensity(edgemap, out_range=(0., 1.))
    merged = dusting * a + edgemap * b

    invmask = merged < maskthreshold
    merged[invmask] = 0

    return merged


def corner_merged(image):
    response_img = corner_fast(image) + corner_kitchen_rosenfeld(image) + corner_shi_tomasi(image) + corner_harris(image)
    response_img = rescale_intensity(response_img)
    return response_img


def detect_corners(image, method="merged"):
    method_choice = {
        "merged": corner_merged,
        "kitchen_rosenfeld": corner_kitchen_rosenfeld,
        "shi_tomasi": corner_shi_tomasi,
        "harris": corner_harris,
        "fast": corner_fast
    }
    try:
        corner_method = method_choice[method]
    except KeyError:
        logger.exception("Invalid corner method: {}".format(method))
        raise
    # Use method with corner_peaks as in jupyter notebook.
    coords = corner_peaks(corner_method(image), min_distance=15, threshold_rel=0.2)
    coords_y, coords_x = zip(*coords)
    coords = np.asarray([list(a) for a in zip(coords_x, coords_y)])
    return coords


def nn_lines_from_corners(coords, distance):
    corner_tree = cKDTree(coords)
    pairs = corner_tree.query_pairs(r=distance)
    lines = [[coords[i], coords[j]] for (i, j) in pairs]
    return lines


def digitise_web(filepath, hough_thresh=20, hough_len=20, hough_gap=5, dustweight=0.5, scharrweight=0.5, mergemaskthresh=0.25, dust_colour=None, return_intermediates=False):
    logger.info("Digitising {}".format(filepath))
    webimg = io.imread(filepath)

    logger.info("Finding image palette...")
    palette, labels, counts = find_image_palette(webimg)
    logger.debug("Palette: {}".format(["#{0:02X}{1:02X}{2:02X}".format(*c.astype(np.uint8)) for c in palette]))

    if dust_colour is None:
        logger.info("Estimating most contrasting colour...")
        contrasting_col_estimate_idx = estimate_contrasting_colour(palette, return_idx=True)
    else:
        logger.info("Finding colour closest to {}...".format(dust_colour))
        contrasting_col_estimate_idx = closest_contrasting_colour(palette, dust_colour, hexcolour=True, return_idx=True)
    logger.debug("Derived dust colour: #{0:02X}{1:02X}{2:02X}".format(*palette[contrasting_col_estimate_idx].astype(np.uint8)))

    logger.info("Isolating dusted pixels...")
    webimg_dusting = np.where(labels == contrasting_col_estimate_idx, 1, 0).reshape(webimg.shape[0:2])

    logger.info("Performing Scharr edge detection...")
    webimg_scharr = filters.scharr(img.color.rgb2gray(webimg))

    logger.info("Merging dusting and scharr images...")
    # May want to provide access to these merge params
    merged = merge_processed_images(webimg_dusting, webimg_scharr, dustweight, scharrweight, mergemaskthresh)

    logger.info("Performing probabilistic hough line transform, thresh={}, len={}, gap={}...".format(hough_thresh, hough_len, hough_gap))
    webimg_hough = img.transform.probabilistic_hough_line(merged, threshold=hough_thresh, line_length=hough_len, line_gap=hough_gap)
    web_dict = {"dimensions": merged.shape[::-1], "lines": webimg_hough, "corners": []}

    logger.info("Digitisation complete.")
    if return_intermediates:
        intermediates = {"palette": palette,
                         "labels": labels,
                         "counts": counts,
                         "contrasting_colour": palette[contrasting_col_estimate_idx],
                         "dusting": webimg_dusting,
                         "scharr": webimg_scharr,
                         "merged": merged}
        return web_dict, intermediates
    else:
        return web_dict


def digitise_web_corner(filepath, dustweight=0.5, scharrweight=0.5, mergemaskthresh=0.25, dust_colour=None, return_intermediates=False):
    logger.info("Digitising {}".format(filepath))
    webimg = io.imread(filepath)

    logger.info("Finding image palette...")
    palette, labels, counts = find_image_palette(webimg)
    logger.debug("Palette: {}".format(["#{0:02X}{1:02X}{2:02X}".format(*c.astype(np.uint8)) for c in palette]))

    if dust_colour is None:
        logger.info("Estimating most contrasting colour...")
        contrasting_col_estimate_idx = estimate_contrasting_colour(palette, return_idx=True)
    else:
        logger.info("Finding colour closest to {}...".format(dust_colour))
        contrasting_col_estimate_idx = closest_contrasting_colour(palette, dust_colour, hexcolour=True, return_idx=True)
    logger.debug("Derived dust colour: #{0:02X}{1:02X}{2:02X}".format(*palette[contrasting_col_estimate_idx].astype(np.uint8)))

    logger.info("Isolating dusted pixels...")
    webimg_dusting = np.where(labels == contrasting_col_estimate_idx, 1, 0).reshape(webimg.shape[0:2])

    logger.info("Performing Scharr edge detection...")
    webimg_scharr = filters.scharr(img.color.rgb2gray(webimg))

    logger.info("Merging dusting and scharr images...")
    # May want to provide access to these merge params
    merged = merge_processed_images(webimg_dusting, webimg_scharr, dustweight, scharrweight, mergemaskthresh)

    # logger.info("Performing probabilistic hough line transform, thresh={}, len={}, gap={}...".format(hough_thresh, hough_len, hough_gap))
    # webimg_hough = img.transform.probabilistic_hough_line(merged, threshold=hough_thresh, line_length=hough_len, line_gap=hough_gap)

    # TODO, FW 2023: HERE we now need to switch into NJA
    # Take merged and skeletonise
    # Find corners a la NJA
    # Trace web lines as above
    
    corners = detect_corners(merged, "kitchen_rosenfeld")
    webimg_lines = nn_lines_from_corners(corners, max(merged.shape[::-1])/10)
    web_dict = {"dimensions": merged.shape[::-1], "lines": webimg_lines, "corners": corners}

    logger.info("Digitisation complete.")
    if return_intermediates:
        intermediates = {"palette": palette,
                         "labels": labels,
                         "counts": counts,
                         "contrasting_colour": palette[contrasting_col_estimate_idx],
                         "dusting": webimg_dusting,
                         "scharr": webimg_scharr,
                         "merged": merged}
        return web_dict, intermediates
    else:
        return web_dict


def digitise_web_nja(
        filepath, alreadybinarised=False,
        multicore=False, try_predict=False, jump=10, lookback=10, max_thresh=30,
        clusterlevel=2):

    logger.info("Digitising {}".format(filepath))
    webimg = io.imread(filepath)
    webimg_grey = rgb2gray(webimg[:, :, :3])
    if not alreadybinarised:
        logger.info("Binarising image...")
        binarised = webimg_grey    # TODO: Add binarisation using u-net
    else:
        binarised = webimg_grey
    logger.info("Loading binarised image into NJA and skeletonising...")
    net = NJA.NJANet(binarised)
    net.skeletonize()
    logger.info("Finding nodes...")
    net.find_nodes()
    logger.info("Finding directions...")
    net.find_directions()
    logger.info("Tracing paths...")
    if multicore:
        net.trace_paths_multicore(try_predict, jump, lookback, max_thresh)
    else:
        net.trace_paths(try_predict, jump, lookback, max_thresh)
    logger.info("Cleaning edges...")
    net.clean_edges()
    if clusterlevel > 0:
        logger.info("Clustering nodes...")
        net.cluster_close_nodes(clusterlevel)
    logger.info("Done.")

    return net


def logtest_digitise():
    s = "Digitise logger"
    logger.critical(s)
    logger.error(s)
    logger.warning(s)
    logger.info(s)
    logger.debug(s)
