"""SAM seed, mask selection and filtering rules from the roof research."""
import numpy as np
from scipy import ndimage


def disk(radius):
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    return x*x+y*y <= radius*radius


def seeds(footprint, count=5):
    distance = ndimage.distance_transform_edt(footprint)
    points = [np.unravel_index(int(np.argmax(distance)), distance.shape)[::-1]]
    ys, xs = np.nonzero(distance > max(distance.max()*0.35, 3))
    if len(xs) > 1:
        idx = np.linspace(0, len(xs)-1, min(count-1, len(xs))).astype(int)
        points.extend((int(xs[i]), int(ys[i])) for i in idx)
    return np.asarray(points[:count], dtype=int)


def negatives(footprint, count=8):
    distance = ndimage.distance_transform_edt(~footprint)
    ys, xs = np.nonzero((distance > 6) & (distance < 20))
    if not len(xs): return np.empty((0, 2), dtype=int)
    indices = np.linspace(0, len(xs)-1, min(count, len(xs))).astype(int)
    return np.asarray([(int(xs[i]), int(ys[i])) for i in indices], dtype=int)


def select_mask(masks, scores, points):
    eligible = [i for i, mask in enumerate(masks)
                if 200 <= mask.sum() <= mask.size*0.85 and mask[points[0, 1], points[0, 0]]]
    index = max(eligible, key=lambda i: scores[i]) if eligible else int(np.argmax(scores))
    return np.asarray(masks[index], dtype=bool)


def refine(mask, footprint, rgb, points, ground_resolution):
    mask = mask & footprint
    if not mask.any(): return mask
    labels, n = ndimage.label(mask)
    ids = {labels[y, x] for x, y in points if labels[y, x] > 0}
    if ids:
        mask = np.isin(labels, list(ids))
    elif n:
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        mask = labels == int(sizes.argmax())
    red, green, blue = [rgb[..., i].astype(np.int16) for i in range(3)]
    gray = rgb.astype(np.float32).mean(axis=2)
    void = mask & ((gray < 78) | ((green-red > 8) & (green-blue > 8)))
    void = ndimage.binary_opening(void, structure=disk(2))
    border = mask ^ ndimage.binary_erosion(mask, structure=disk(3))
    labels, n = ndimage.label(void)
    if not n: return mask
    touching = set(np.unique(labels[border])) - {0}
    sizes = np.bincount(labels.ravel(), minlength=n+1)
    keep = [i for i in range(1, n+1) if i not in touching and sizes[i]*ground_resolution**2 >= 25]
    return mask & ~np.isin(labels, keep) if keep else mask


def exclusions(rgb, roof):
    gray = rgb.astype(np.float32).mean(axis=2)
    mean = ndimage.uniform_filter(gray, 5)
    mean_sq = ndimage.uniform_filter(gray*gray, 5)
    rough = np.sqrt(np.maximum(mean_sq-mean*mean, 0)) > 12
    usable = ndimage.binary_opening(roof & ~((gray < 60) | rough), structure=disk(1))
    return roof & ~usable
