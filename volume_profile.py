"""Bar-approximated volume profile from one-minute OHLCV. This is not traded volume at price.

Minute bars do not show how much volume traded at each price. Every profile here spreads
each minute's volume uniformly over that minute's [low, high], so results are labeled
PROFILE_METHOD wherever they appear. It is one specified convention, not a promise to
match any charting platform, and sparse bars or a different bin count can move its levels.
"""

import numpy as np

PROFILE_METHOD = "bar_approximated_volume_profile"
VALUE_AREA_FRACTION = 0.70


def allocate(low, high, volume, edges):
    """Volume per bin, splitting each bar in proportion to its overlap with that bin.

    Bins are left-inclusive and right-exclusive, except the last bin, which also includes
    its right edge. A bar with high == low puts all of its volume in the bin that contains
    that price.
    """
    low, high, volume = (np.asarray(x, dtype=float) for x in (low, high, volume))
    edges = np.asarray(edges, dtype=float)
    n = len(edges) - 1
    out = np.zeros(n)
    point = high == low
    if point.any():
        idx = np.clip(np.searchsorted(edges, low[point], side="right") - 1, 0, n - 1)
        np.add.at(out, idx, volume[point])
    if (~point).any():
        lo, hi, v = low[~point, None], high[~point, None], volume[~point, None]
        overlap = np.clip(np.minimum(hi, edges[None, 1:]) - np.maximum(lo, edges[None, :-1]), 0, None)
        out += (v * overlap / (hi - lo)).sum(axis=0)
    return out


def bar_profile(low, high, volume, n_bins):
    """(edges, bin_volume) using n_bins equal bins from the lowest low to the highest high.

    Returns None if the input cannot support a profile: no bars, non-finite or non-positive
    prices, high < low, negative or non-finite volume, zero total volume, or a zero range.
    """
    low, high, volume = (np.asarray(x, dtype=float) for x in (low, high, volume))
    if not len(low) or not (np.isfinite(low).all() and np.isfinite(high).all() and np.isfinite(volume).all()):
        return None
    if (low <= 0).any() or (high < low).any() or (volume < 0).any():
        return None
    total = volume.sum()
    if not (high.max() > low.min() and total > 0):
        return None
    edges = np.linspace(low.min(), high.max(), n_bins + 1)
    bins = allocate(low, high, volume, edges)
    if not np.isclose(bins.sum(), total, rtol=1e-9, atol=0):  # the allocation must conserve volume
        raise RuntimeError(f"profile allocated {bins.sum()} of {total} volume")
    return edges, bins


def value_area(bins, edges, fraction=VALUE_AREA_FRACTION):
    """(VAL, POC, VAH) for one profile.

    POC is the center of the highest-volume bin, and ties choose the lower-priced bin. The
    contiguous value area starts at the POC bin. It repeatedly adds whichever adjacent bin
    has more volume, taking the lower bin on ties and the remaining side once one edge is
    exhausted, until it holds at least `fraction` of the volume. VAL and VAH are the outer
    edges of that area.
    """
    bins = np.asarray(bins, dtype=float)
    poc = int(np.argmax(bins))  # first maximum = lowest price
    lo = hi = poc
    included, target = bins[poc], fraction * bins.sum()
    while included < target and (lo > 0 or hi < len(bins) - 1):
        take_lower = hi == len(bins) - 1 or (lo > 0 and bins[lo - 1] >= bins[hi + 1])
        if take_lower:
            lo -= 1
            included += bins[lo]
        else:
            hi += 1
            included += bins[hi]
    return edges[lo], (edges[poc] + edges[poc + 1]) / 2, edges[hi + 1]


def low_volume_nodes(bins):
    """Bin indices of low-volume nodes, in price order.

    Volume is smoothed with the centered [1, 2, 1] / 4 kernel wherever the full kernel fits,
    which means every bin except the first and last. The first two and last two bins are
    never candidates. A node has positive raw volume, smoothed volume strictly below both
    immediate neighbors, and smoothed volume no more than half the smaller of the largest
    smoothed volumes on its left and on its right.
    """
    bins = np.asarray(bins, dtype=float)
    n = len(bins)
    smooth = np.full(n, np.nan)
    smooth[1:-1] = (bins[:-2] + 2 * bins[1:-1] + bins[2:]) / 4
    nodes = []
    for i in range(2, n - 2):
        left, right = smooth[1:i].max(), smooth[i + 1:n - 1].max()
        if (bins[i] > 0 and smooth[i] < smooth[i - 1] and smooth[i] < smooth[i + 1]
                and smooth[i] <= 0.5 * min(left, right)):
            nodes.append(i)
    return nodes
