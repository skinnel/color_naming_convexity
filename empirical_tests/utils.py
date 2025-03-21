"""Utility functions to aid in running the empirical_tests """

import numpy as np
from scipy.spatial import ConvexHull


def point_in_hull(point, hull, tolerance=1e-12):
    """A function to determine if the provide point lies within the provided convex hull. The code and reasoning are
    taken from https://stackoverflow.com/a/42165596.

    Arguments
    ---------
    point
        A point in R^n denoted with standard notation.
    hull
        A pre-defined convex hull determined from a series of points in R^n.

    Returns
    -------
    True if the point is inside or along the boundary of the convex hull, False otherwise.

    """
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)

def get_level_convexity(x_coordinates: np.array,
                        x_probs: np.array,
                        level_p: float,
                        min_pts: int = 3) -> int:
    """ A function to calculate the percentage of the convex hull that is covered for a given level of the domain, where
    the level is defined to be all x values with a probability equal to level_p or greater.

    Notes
    -----
    We will find all points in the domain with an affiliated probability greater-than-or-equal-to level_p and then see
    how many additional points from the domain are needed to fill in the convex hull. i.e. we will determine which
    points with probability less-than level_p are "in-between" (in a Euclidean sense) 2+ points that have probability
    greater-than-or-equal-to level_p. The union of these two sets of points will form the convex hull of the level. We
    will return the tuple (number of points with probability >= level_p, number of points in the convex hull)

    Parameters
    ----------
    x_coordinates
        A 2D numpy array that contains the coordinates of each point in the domain.
    x_probs
        A 1D numpy array that contains the probability affiliated with a given point in the domain. Its order matches
        that found in x_coordinates.
    level_p
        A decimal value indicating the level to be investigated.
    min_pts
        The minimum number of points required to construct a convex hull.

    Returns
    -------
    level_ct
        indicates how many points are in the level set
    hull_ct
        indicates how many points are in the convex hull
    """

    # Get the points in the level set
    level_set = x_coordinates[np.nonzero(x_probs >= level_p)[0], :]
    outside_level = x_coordinates[np.nonzero(x_probs < level_p)[0], :]
    level_ct = level_set.shape[0]

    # Get the points in the convex hull
    try:
        hull = ConvexHull(level_set)
        hull_ct = level_ct
        for x_i in range(outside_level.shape[0]):
            x = outside_level[x_i, :]
            if point_in_hull(x, hull):
                hull_ct += 1
    except Exception:
        level_ct = level_set.shape[0]
        hull_ct = level_ct

    return level_ct, hull_ct


def get_quasi_concavity_measure(x_coordinates: np.array,
                                    x_probs: np.array,
                                    mesh: float = None) -> float:
    """ A function to calculate the "quasi-concavity measure" that we define to be a normalized summation of the percent
    coverage of the convex hull at each level set.

    Notes
    -----
    We will use the helper function get_level_convexity to find the coverage of the convex hull at each level set. Level
    sets are determined by the mesh parameter (see below for details). Once the coverage of every level set has been
    calculated we take the average across all level sets, resulting in our quasi-concavity metric. Values closer to 1
    indicate that the pdf is "more quasi-concave" and values closer to 0 indicate that the pdf is "less quasi-concave".

    Parameters
    ----------
    x_coordinates
        A 2D numpy array that contains the coordinates of each point in the domain.
    x_probs
        A 1D numpy array that contains the probability affiliated with a given point in the domain. Its order matches
        that found in x_coordinates.
    mesh
        A float value indicating the size of jump between probability values that should be used to specify the level
        sets. If mesh is None then we use a dynamic mesh value so that we calculate a different level set for each
        observed probability value in the data provided. This case requires using a weighted average to calculate the
        final metric, where weights are proportional to the dynamic mesh values.

    Returns
    -------
    qc_metric
        The average convex hull coverage for all the level sets in the pdf.
    """

    # Get bounds on probability values
    pmax = x_probs.max()
    pmin = x_probs.min()

    # Get convex hull coverage for each level set
    hull_pct = []
    if mesh:
        p = pmax
        while p > pmin:
            p = p - mesh
            try:
                level_ct, hull_ct = get_level_convexity(x_coordinates, x_probs, p)
                hull_pct.append(level_ct / hull_ct)
            except:
                hull_pct.append(0)

        hull_pct = np.array(hull_pct)
        qc_metric = np.mean(hull_pct)
    else:
        wts = []
        sorted_p = np.flip(np.unique(x_probs))
        for i in range(2, len(sorted_p)-1):
            p0 = sorted_p[i]
            p1 = sorted_p[i+1]
            mesh = p0 - p1
            #TODO: Temp fix - need to figure out what to do when rank(points defining hull) is too small
            try:
                level_ct, hull_ct = get_level_convexity(x_coordinates, x_probs, p1)
                wts.append(mesh)
                hull_pct.append(level_ct / hull_ct)
            except:
                wts.append(0)
                hull_pct.append(0)
        hull_pct = np.array(hull_pct)
        wts = np.array(wts)
        wts = wts/wts.sum()
        qc_metric = sum(hull_pct * wts)

    return qc_metric