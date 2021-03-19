import numpy as np
# weird import style to un-confuse PyCharm
try:
    from cv2 import cv2
except ImportError:
    import cv2

# The functions:
#  - color_transfer
#  - image_stats
#  - _min_max_scale
#  - _scale_array
#
# were based on https://github.com/jrosebr1/color_transfer/
# and are used and modified here under their original license:
#
# The MIT License (MIT)
#
# Copyright (c) 2014 Adrian Rosebrock, http://www.pyimagesearch.com
# Copyright (c) 2018 Scott White
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ---END LICENSE---
#
# All other code is distributed under the cv2_extras project license


def color_transfer(ref_img, target_img, clip=True, preserve_paper=True):
    """
    Transfers the color distribution from the reference to the target
    image using the mean and standard deviations of the L*a*lab_b*
    color space.

    This implementation is (loosely) based on to the "Color Transfer
    between Images" paper by Reinhard et al., 2001.

    Parameters:
    -------
    ref_img: NumPy array
        OpenCV image in BGR color space (the reference image)
    target_img: NumPy array
        OpenCV image in BGR color space (the target image)
    clip: Should components of L*a*lab_b* image be scaled by np.clip before
        converting back to BGR color space?
        If False then components will be min-max scaled appropriately.
        Clipping will keep target image brightness truer to the input.
        Scaling will adjust image brightness to avoid washed out portions
        in the resulting color transfer that can be caused by clipping.
    preserve_paper: Should color transfer strictly follow methodology
        laid out in original paper? The method does not always produce
        aesthetically pleasing results.
        If False then L*a*lab_b* components will scaled using the reciprocal of
        the scaling factor proposed in the paper.  This method seems to produce
        more consistently aesthetically pleasing results

    Returns:
    -------
    transfer: NumPy array
        OpenCV image (w, h, 3) NumPy array (8-bit unsigned integer)
    """
    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # compute color statistics for the reference and target images
    (lMeanRef, lStdRef, aMeanRef, aStdRef, bMeanRef, bStdRef) = _lab_image_stats(ref_img)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = _lab_image_stats(target_img)

    # subtract the means from the target image
    (light, lab_a, lab_b) = cv2.split(target_img)
    light -= lMeanTar
    lab_a -= aMeanTar
    lab_b -= bMeanTar

    if preserve_paper:
        # scale by the standard deviations using paper proposed factor
        light = (lStdTar / lStdRef) * light
        lab_a = (aStdTar / aStdRef) * lab_a
        lab_b = (bStdTar / bStdRef) * lab_b
    else:
        # scale by the standard deviations using reciprocal of paper proposed factor
        light = (lStdRef / lStdTar) * light
        lab_a = (aStdRef / aStdTar) * lab_a
        lab_b = (bStdRef / bStdTar) * lab_b

    # add in the reference mean
    light += lMeanRef
    lab_a += aMeanRef
    lab_b += bMeanRef

    # clip/scale the pixel intensities to [0, 255] if they fall
    # outside this range
    light = _scale_array(light, clip=clip)
    lab_a = _scale_array(lab_a, clip=clip)
    lab_b = _scale_array(lab_b, clip=clip)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([light, lab_a, lab_b])
    transfer = cv2.cvtColor(transfer.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # return the color transferred image
    return transfer


def _lab_image_stats(image):
    """
    Parameters:
    -------
    image: NumPy array
        OpenCV image in L*a*b* color space

    Returns:
    -------
    Tuple of mean and standard deviations for the L*, a*, and b*
    channels, respectively
    """
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return lMean, lStd, aMean, aStd, bMean, bStd


def _min_max_scale(arr, new_range=(0, 255)):
    """
    Perform min-max scaling to a NumPy array

    Parameters:
    -------
    arr: NumPy array to be scaled to [new_min, new_max] range
    new_range: tuple of form (min, max) specifying range of
        transformed array

    Returns:
    -------
    NumPy array that has been scaled to be in
    [new_range[0], new_range[1]] range
    """
    # get array's current min and max
    mn = arr.min()
    mx = arr.max()

    # check if scaling needs to be done to be in new_range
    if mn < new_range[0] or mx > new_range[1]:
        # perform min-max scaling
        scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
    else:
        # return array if already in range
        scaled = arr

    return scaled


def _scale_array(arr, clip=True):
    """
    Trim NumPy array values to be in [0, 255] range with option of
    clipping or scaling.

    Parameters:
    -------
    arr: array to be trimmed to [0, 255] range
    clip: should array be scaled by np.clip? if False then input
        array will be min-max scaled to range
        [max([arr.min(), 0]), min([arr.max(), 255])]

    Returns:
    -------
    NumPy array that has been scaled to be in [0, 255] range
    """
    if clip:
        scaled = np.clip(arr, 0, 255)
    else:
        scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
        scaled = _min_max_scale(arr, new_range=scale_range)

    return scaled
