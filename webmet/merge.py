import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import json
import itertools
import multiprocessing as mp
import sys
from webmet.exceptions import MergeError
import logging
logger = logging.getLogger(__name__)
logger.propagate = True


def line_length(line):
    """Find length of a line given two coordinates.
        >>> line_length([[0,0],[3,4]])
        5.0
    """
    return ((line[0][0] - line[1][0]) ** 2 + (line[0][1] - line[1][1]) ** 2) ** 0.5


def find_centroid(line1, line2):
    """
        >>> find_centroid([[0,0],[0,4]],[[4,0],[4,4]])
        (2.0, 2.0)
    """
    li = line_length(line1)
    lj = line_length(line2)

    xg = (li * (line1[0][0] + line1[1][0]) + lj * (line2[0][0] + line2[1][0])) / (2 * (li + lj))
    yg = (li * (line1[0][1] + line1[1][1]) + lj * (line2[0][1] + line2[1][1])) / (2 * (li + lj))
    return (xg, yg)


def multi_find_centroid(lines):
    """
        >>> multi_find_centroid([[[0, 0], [0, 4]], [[4, 0], [4, 4]], [[0, 4], [4, 4]], [[0, 0], [4, 0]]])
        (2.0, 2.0)
    """
    ls = [line_length(l) for l in lines]
    xs, ys = zip(*[(ls[idx] * (l[0][0] + l[1][0]), ls[idx] * (l[0][1] + l[1][1])) for idx, l in enumerate(lines)])

    xg = (sum(xs)) / (2 * (sum(ls)))
    yg = (sum(ys)) / (2 * (sum(ls)))
    return (xg, yg)


def find_line_orientation(line):
    """get orientation of a line
    https://en.wikipedia.org/wiki/Atan2
        >>> find_line_orientation([[0,0],[-4,0]])
        0.0
    """
    return np.arctan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))


def point_to_framespace(point, centroid, orientation):
    dxg = (point[1] - centroid[1]) * np.sin(orientation) + (point[0] - centroid[0]) * np.cos(orientation)
    dyg = (point[1] - centroid[1]) * np.cos(orientation) - (point[0] - centroid[0]) * np.sin(orientation)
    return (dxg, dyg)


def lines_to_framespace(lines, centroid, orientation):
    return [[point_to_framespace(point, centroid, orientation) for point in line] for line in lines]


def point_from_framespace(point, centroid, orientation):
    dx = centroid[0] + np.cos(orientation) * point[0] - np.sin(orientation) * point[1]
    dy = centroid[1] + np.sin(orientation) * point[0] + np.cos(orientation) * point[1]
    return (dx, dy)


def mergable(line1, line2, threshold=np.pi / 20):
    """Check if two lines can be merged. Default threshold provides 10% coverage (maybe 20%)"""
    if line1[0] == line1[1] or line2[0] == line2[1]:
        raise ValueError("Lines cannot have the same start and end coordinates.")
    if line1[0] in line2 and line1[1] in line2:
        return False
    orientation_delta = find_orientation_difference(line1, line2)
    if orientation_delta <= threshold:
        return True
    elif (np.pi - orientation_delta) <= threshold:
        return True
    else:
        return False


def find_orientation_difference(line1, line2, absolute=True):
    """Find the smallest orientation difference between two lines, """
    if isinstance(line1, WebLine):
        l1_theta = line1.orientation
    else:
        l1_theta = find_line_orientation(line1)
    if isinstance(line2, WebLine):
        l2_theta = line2.orientation
    else:
        l2_theta = find_line_orientation(line2)

    a = (l1_theta - l2_theta) % (np.pi * 2)
    b = (l2_theta - l1_theta) % (np.pi * 2)
    #     print(a)
    #     print(b)
    out = -a if a < b else b
    if absolute:
        return abs(out)
    else:
        return out


class WebKernel:
    def __init__(self, webdict=None):
        if webdict:
            self.webdict = webdict
            self.lines = [WebLine(x) for x in self.webdict["lines"]]
            self.dimensions = self.webdict.get("dimensions", self.find_dimensions())
            self.remove_zero_lines()
        else:
            self.webdict = dict()
            self.lines = []
            self.dimensions = []

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        return self.lines[i]

    def __str__(self):
        if len(self.lines):
            return "Web Kernel of {0} lines, ({1}x{2})".format(len(self.lines), *self.dimensions)
        else:
            return "Empty Web Kernel"

    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__, self.webdict)

    def find_dimensions(self):
        """Find minimum dimensions of a web kernel"""
        flattened = [item for sublist in self.lines for item in sublist]
        xs, ys = zip(*flattened)
        dimx = int(max(xs))
        dimy = int(max(ys))
        return [dimx, dimy]

    def remove_zero_lines(self):
        self.lines = [x for x in self.lines if x.length > 0]

    def find_merge_candidates(self):
        candidates = list(itertools.combinations(self.lines, 2))
        print(len(candidates))
        candidates = [c for c in candidates if mergable(*c)]
        print(len(candidates))
        return candidates

    def to_polar(self, origin=None, flipped=False):
        return [l.to_polar(origin, flipped) for l in self.lines]

    def ellipse_transform(self, origin, orientation, scaling):
        return [l.ellipse_transform(origin, orientation, scaling) for l in self.lines]

    def ellipse_and_polar(self, origin, orientation, scaling, flipped=False):
        return [l.ellipse_and_polar(origin, orientation, scaling, flipped) for l in self.lines]

    def as_dict(self):
        """Export a Web Kernel as a dictionary"""
        return {"dimensions": self.dimensions, "lines": [line.export() for line in self.lines]}


class WebLine:
    def __init__(self, line, orientation=None, length=None):
        self.line = line

        if orientation is not None:
            self.orientation = orientation
        else:
            self.orientation = self.find_orientation()

        if length is not None:
            self.length = length
        else:
            self.length = self.find_length()

    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__, self.line)

    def __str__(self):
        return "{0}\nLength: {1}\nOrientation: {2}".format(self.line, self.length, self.orientation)

    # Implement getitem to allow indexing the line to get out the points easily
    def __getitem__(self, i):
        return self.line[i]

    #     def __len__(self):
    #         return len(self.line)

    def find_orientation(self):
        return find_line_orientation(self.line)

    def find_length(self):
        return line_length(self.line)

    def to_polar(self, origin=None, flipped=False, points=None):
        if points is None:
            points = self.line
        if isinstance(points, list):
            points = np.array(points)
        x, y = points[:, 0], points[:, 1]
        if origin is not None:
            ox, oy = origin
            x = x - ox
            y = y - oy
        r = np.sqrt(x ** 2 + y ** 2)
        t = np.arctan2(y, x)
        if flipped:
            return list(zip(t, r))
        return list(zip(r, t))

    def ellipse_transform(self, origin, orientation, scaling):
        points = self.line
        if isinstance(points, list):
            points = np.array(points)

        x, y = points[:, 0], points[:, 1]

        tx = scaling * (((y - origin[1]) * np.sin(orientation)) + ((x - origin[0]) * np.cos(orientation)))
        ty = (1 / scaling) * (((y - origin[1]) * np.cos(orientation)) - ((x - origin[0]) * np.sin(orientation)))

        # Backtransform
        newx = (origin[0] + np.cos(orientation) * tx - np.sin(orientation) * ty)#.tolist()
        newy = (origin[1] + np.sin(orientation) * tx + np.cos(orientation) * ty)#.tolist()
        return list(zip(newx, newy))

    def ellipse_and_polar(self, origin, orientation, scaling, flipped=False):
        return self.to_polar(origin, flipped, points=self.ellipse_transform(origin, orientation, scaling))

    def export(self):
        return self.line


def multi_merge_line_segments(lines, dmax_x=None, dmax_y=None, dmax_y_o=None, dmax_y_p=None):
    """Merge multiple line segments using a genericised version of the Tavares & Padliha algorithm"""
    centroid = multi_find_centroid(lines)

    #     lrmin = 50

    # Find orientation
    thetas = []
    lengths = []
    for line in lines:
        if isinstance(line, WebLine):
            thetas.append(float(line.orientation))
            lengths.append(line.length)
        else:
            thetas.append(float(find_line_orientation(line)))
            lengths.append(line_length(line))

    #     normalised_thetas_old = [t - min(thetas) for t in thetas]
    normalised_thetas = [(t - min(thetas)) - 2 * np.pi if t - min(thetas) > np.pi else t - min(thetas) for t in thetas]
    #     corrected_thetas = [thetas[idx] if abs(t) <= np.pi/2 else thetas[idx] - np.pi*(thetas[idx]/abs(thetas[idx])) for idx, t in enumerate(normalised_thetas)]
    corrected_thetas = []
    for idx, t in enumerate(thetas):
        t_orig = thetas[idx]
        if abs(t) <= np.pi / 2 or t_orig == 0:
            corrected_thetas.append(t_orig)
        else:
            corrected_thetas.append(t_orig - np.pi * (t_orig / abs(t_orig)))
    #     print(corrected_thetas)
    #     print(normalised_thetas_old)

    thetar = sum([lengths[idx] * t for idx, t in enumerate(corrected_thetas)]) / sum(lengths)

    #     print(thetar)

    #     return 0

    # Transform lines to framespace
    transformed_lines = lines_to_framespace(lines, centroid, thetar)
    # ---
    # Now merge to one list of coordinates
    transformed_points = [item for sublist in transformed_lines for item in sublist]

    xs, ys = zip(*transformed_points)
    minx = int(min(xs))
    maxx = int(max(xs))
    #     miny = int(min(ys))
    #     maxy = int(max(ys))
    #     l1_X_G = abs(transformed_lines[0][1][0] - transformed_lines[0][0][0])
    #     l2_X_G = abs(transformed_lines[1][1][0] - transformed_lines[1][0][0])

    lr = maxx - minx

    # #     if lr < lrmin:
    # #         raise MergeError("Line too small.")

    #     # Set working dmax_y value to the same as dmax_y. If dmax_y is also None then it doesn't matter.
    #     dmax_y_working = dmax_y
    # #     print("dmax_x = {}".format(dmax_x))

    #     if lr >= (l1_X_G + l2_X_G):
    #         # Not Overlapping!
    # #         print("Not Overlapping!")
    #         # Test dmax_x only in the case that candidate lines are not overlapping
    #         if dmax_x is not None:
    #             if (lr - (l1_X_G + l2_X_G)) > dmax_x:
    #                 raise MergeError("X distance outside maximum threshold value. {} > {}".format((lr - (l1_X_G + l2_X_G)), dmax_x))

    #     elif np.isclose(lr, l1_X_G) or np.isclose(lr, l2_X_G):
    #         # Here we use isclose rather than == due to minor rounding errors when transforming to framespace
    #         # Fully Overlapping!
    # #         print("Fully Overlapping!")
    #         # If dmax_y for overlapping has been specified, use that, otherwise leave as normal dmax_y
    #         if dmax_y_o is not None:
    #             dmax_y_working = dmax_y_o

    #     elif lr < (l1_X_G + l2_X_G):
    #         # Partially Overlapping!
    # #         print("Partially Overlapping!")
    #         # If dmax_y for partial has been specified, use that, otherwise leave as normal dmax_y
    #         if dmax_y_p is not None:
    #             dmax_y_working = dmax_y_p
    #     else:
    #         print("Inconclusively Overlapping.")
    #         raise MergeError("Overlap of lines could not be conclusively evaluated.")

    # #     print("dmax_x = {}, dmax_y = {}".format(dmax_x, dmax_y_working))
    #     if dmax_y_working is not None:
    #         if abs(maxy - miny) > dmax_y_working:
    #             raise MergeError("Y distance outside maximum threshold value. {} > {}".format(abs(maxy - miny), dmax_y_working))

    # Retransform min and max values back to normal space
    merged_line = WebLine(
        [point_from_framespace((minx, 0), centroid, thetar), point_from_framespace((maxx, 0), centroid, thetar)],
        thetar, lr)
    #     merged_line = (point_from_framespace((minx, 0), centroid, thetar),point_from_framespace((maxx, 0), centroid, thetar))
    return merged_line


def merge_line_segments(line1, line2, dmax_x=None, dmax_y=None, dmax_y_o=None, dmax_y_p=None):
    """Merge line segments using the Tavares & Padliha algorithm"""
    centroid = find_centroid(line1, line2)

    # Find orientation
    li = line_length(line1)
    lj = line_length(line2)
    thetai = float(find_line_orientation(line1))
    thetaj = float(find_line_orientation(line2))

    if abs(thetai - thetaj) <= (np.pi / 2):
        thetar = (li * thetai + lj * thetaj) / (li + lj)
    else:
        if thetaj == 0:
            # If thetaj is 0 then assume that 0/0 == 1, rather than nan or inf.
            # This could be tricky in the scenario that thetaj = -0
            thetar = (li * thetai + lj * (thetaj - np.pi)) / (li + lj)
        else:
            thetar = (li * thetai + lj * (thetaj - np.pi * (thetaj / abs(thetaj)))) / (li + lj)

    # Transform lines to framespace
    transformed_lines = lines_to_framespace((line1, line2), centroid, thetar)
    # ---
    # Now merge to one list of coordinates
    transformed_points = [item for sublist in transformed_lines for item in sublist]

    xs, ys = zip(*transformed_points)
    minx = int(min(xs))
    maxx = int(max(xs))
    #     miny = int(min(ys))
    #     maxy = int(max(ys))
    #     l1_X_G = abs(transformed_lines[0][1][0] - transformed_lines[0][0][0])
    #     l2_X_G = abs(transformed_lines[1][1][0] - transformed_lines[1][0][0])

    lr = maxx - minx

    # Set working dmax_y value to the same as dmax_y. If dmax_y is also None then it doesn't matter.
    #     dmax_y_working = dmax_y

    #     if lr >= (l1_X_G + l2_X_G):
    #         # Not Overlapping!
    #         # Test dmax_x only in the case that candidate lines are not overlapping
    #         if dmax_x is not None:
    #             if (lr - (l1_X_G + l2_X_G)) > dmax_x:
    #                 raise MergeError("X distance outside maximum threshold value. {} > {}".format((lr - (l1_X_G + l2_X_G)), dmax_x))

    #     elif np.isclose(lr, l1_X_G) or np.isclose(lr, l2_X_G):
    #         # Here we use isclose rather than == due to minor rounding errors when transforming to framespace
    #         # Fully Overlapping!
    #         # If dmax_y for overlapping has been specified, use that, otherwise leave as normal dmax_y
    #         if dmax_y_o is not None:
    #             dmax_y_working = dmax_y_o

    #     elif lr < (l1_X_G + l2_X_G):
    #         # Partially Overlapping!
    #         # If dmax_y for partial has been specified, use that, otherwise leave as normal dmax_y
    #         if dmax_y_p is not None:
    #             dmax_y_working = dmax_y_p
    #     else:
    #         print("Inconclusively Overlapping.")
    #         raise MergeError("Overlap of lines could not be conclusively evaluated.")

    #     if dmax_y_working is not None:
    #         if abs(maxy - miny) > dmax_y_working:
    #             raise MergeError("Y distance outside maximum threshold value. {} > {}".format(abs(maxy - miny), dmax_y_working))

    # Retransform min and max values back to normal space
    merged_line = (point_from_framespace((minx, 0), centroid, thetar), point_from_framespace((maxx, 0), centroid, thetar))
    return merged_line


def load_kernel(kernelpath):
    with open(os.path.join(kernelpath), "r") as f:
        web_dict = json.load(f)
    return WebKernel(web_dict)


def logtest_merge():
    s = "Merge logger"
    logger.critical(s)
    logger.error(s)
    logger.warning(s)
    logger.info(s)
    logger.debug(s)
