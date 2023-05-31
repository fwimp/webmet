import numpy as np
from itertools import chain

def is_vert(theta, threshold=np.pi/20):
    if np.pi/2 + threshold >= theta >= np.pi/2 - threshold:
        return True
    if -np.pi/2 + threshold >= theta >= -np.pi/2 - threshold:
        return True
    return False

def is_horiz(theta, threshold=np.pi/20):
    if 0 + threshold >= theta >= 0 - threshold:
        return True
    if -np.pi + threshold >= theta >= -np.pi:
        return True
    if np.pi >= theta >= np.pi - threshold:
        return True
    return False


def classify_threads_flexible(kernel, hub, threshold=np.pi / 20, hub_threshold_prop=0.07, ellipse_orientation=0, ellipse_scale=1):
    """Perform polar coordinate unwrap, rescale, and assessment.
    By default, return proportion of lines left uncategorised."""

    if ellipse_scale != 1:
        polar = kernel.ellipse_and_polar(origin=hub, orientation=ellipse_orientation, scaling=ellipse_scale,
                                         flipped=True, )
    else:
        polar = kernel.to_polar(origin=hub, flipped=True)

    # Rescale to the size of the web kernel

    dims = max(polar.find_dimensions(polar))

    polar.rescale(dimensions=kernel.dimensions).recalculate_transformed_orientations()

    t, r = zip(*list(chain.from_iterable(polar.as_list_transformed())))
    rmax = max(r)
    # print(rmax * hub_threshold_prop)
    rmin = min(r)

    out = dict()
    for l in polar:
        if is_vert(l.transformed_orientation, threshold):
            out[l.id] = 1  # Radial Line
        else:
            #         elif is_horiz(l.transformed_orientation, threshold):
            l_t, l_r = np.array(l.transformed_line)[:, 0], np.array(l.transformed_line)[:, 1]
            if any(l_r <= rmax * hub_threshold_prop):
                out[l.id] = 2  # Hub
            else:
                out[l.id] = 3  # Sticky
    return out
