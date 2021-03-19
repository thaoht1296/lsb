import os
import numpy as np
from scipy import optimize
from skimage.segmentation import slic
import matplotlib.pyplot as plt

# weird import style to un-confuse PyCharm
try:
    from cv2 import cv2
except ImportError:
    import cv2

block_strel = np.ones((3, 3))
ellipse_strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
ellipse90_strel = np.rot90(ellipse_strel)
circle_strel = np.bitwise_or(ellipse_strel, ellipse90_strel)
cross_strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def calculate_distance(x1, y1, x2, y2):
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def get_bounding_rect(contour):
    b_rect = cv2.boundingRect(contour)
    x1 = b_rect[0]
    x2 = b_rect[0] + b_rect[2]
    y1 = b_rect[1]
    y2 = b_rect[1] + b_rect[3]

    return x1, y1, x2, y2


def crop_image(img, x1, y1, x2, y2):

    # crop region and poly points for efficiency
    crop_img = img[y1:y2, x1:x2]

    return crop_img


def save_image(base_dir, img_name, rgb_img):
    image_path = os.path.join(
        base_dir,
        img_name
    )
    cv2.imwrite(
        image_path,
        cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    )


def get_flat_hsv_channels(hsv_img, mask=None):
    """
    Returns flattened hue, saturation, and values from given HSV image.
    """
    hue = hsv_img[:, :, 0].flatten()
    sat = hsv_img[:, :, 1].flatten()
    val = hsv_img[:, :, 2].flatten()

    if mask is not None:
        flat_mask = mask.flatten()

        hue = hue[flat_mask > 0]
        sat = sat[flat_mask > 0]
        val = val[flat_mask > 0]

    return hue, sat, val


def fill_holes(mask):
    """
    Fills holes in a given binary mask.
    """
    ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(new_mask, contours, -1, 255, -1)

    return new_mask


def plot_contours(img_hsv, contours):
    new_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    cv2.drawContours(new_img, contours, -1, (0, 255, 0), 5)
    plt.figure(figsize=(16, 16))
    plt.imshow(new_img)
    plt.show()


def translate_contour(contour, x, y):
    if len(contour.shape) == 3:
        # dealing with OpenCV type contour
        contour[:, :, 0] = contour[:, :, 0] - x
        contour[:, :, 1] = contour[:, :, 1] - y
    else:
        # assume a simple array of x, y coordinates
        contour[:, 0] = contour[:, 0] - x
        contour[:, 1] = contour[:, 1] - y

    return contour


def smooth_contours(contours, peri_factor=0.007):
    smoothed_contours = []

    for c in contours:
        peri = cv2.arcLength(c, True)
        smooth_c = cv2.approxPolyDP(c, peri_factor * np.sqrt(peri), True)

        smoothed_contours.append(smooth_c)

    return smoothed_contours


def filter_contours_by_size(mask, min_size=1024, max_size=None):
    ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if max_size is None:
        max_size = mask.shape[0] * mask.shape[1]
    min_size = min_size

    good_contours = []

    for c in contours:
        rect = cv2.boundingRect(c)
        rect_area = rect[2] * rect[3]

        if max_size >= rect_area >= min_size:
            good_contours.append(c)

    return good_contours


def find_border_contours(contours, img_h, img_w):
    """
    Given a list of contours, splits them into 2 lists: the border contours and
    non-border contours

    Args:
        contours: list of contours to separate
        img_h: original image height
        img_w: original image width

    Returns:
        2 lists, the first being the border contours

    Raises:
        tbd
    """

    min_y = 0
    min_x = 0

    max_y = img_h - 1
    max_x = img_w - 1

    mins = {min_x, min_y}
    maxs = {max_x, max_y}

    border_contours = []
    non_border_contours = []

    for c in contours:
        rect = cv2.boundingRect(c)

        c_min_x = rect[0]
        c_min_y = rect[1]
        c_max_x = rect[0] + rect[2] - 1
        c_max_y = rect[1] + rect[3] - 1

        c_mins = {c_min_x, c_min_y}
        c_maxs = {c_max_x, c_max_y}

        if len(mins.intersection(c_mins)) > 0 or len(maxs.intersection(c_maxs)) > 0:
            border_contours.append(c)
        else:
            non_border_contours.append(c)

    return border_contours, non_border_contours


def fill_border_contour(contour, img_shape):
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, cv2.FILLED)

    # Extract the perimeter pixels, leaving out the last pixel
    # of each side as it is included in the next side (going clockwise).
    # This makes all the side arrays the same length.
    # We also flip the bottom and left side, as we want to "unwrap" the
    # perimeter pixels in a clockwise fashion.
    top = mask[0, :-1]
    right = mask[:-1, -1]
    bottom = np.flipud(mask[-1, 1:])
    left = np.flipud(mask[1:, 0])

    # combine the perimeter sides into one continuous array
    perimeter_pixels = np.concatenate([top, right, bottom, left])

    region_boundary_locs = np.where(perimeter_pixels == 255)[0]

    # the perimeter here is not a geometric perimeter but the number of pixels around the image
    img_h = img_shape[0]
    img_w = img_shape[1]
    perimeter = (img_h - 1) * 2 + (img_w - 1) * 2

    # account for the wrap around from the last contour pixel to the end,
    # i.e. back at the start at (0, 0)
    wrap_distance = region_boundary_locs.max() - perimeter

    # insert the wrap distance in front of the region boundary locations
    region_boundary_locs = np.concatenate([[wrap_distance], region_boundary_locs])

    # calculate the gap size between boundary pixel locations
    gaps = np.diff(region_boundary_locs)

    # if there's only one gap, the contour is already filled
    if not np.sum(gaps > 1) > 1:
        return mask

    # add one to the results because of the diff offset
    max_gap_idx = np.where(gaps == gaps.max())[0] + 1

    # there should only be one, else we have a tie and should probably ignore that case
    if max_gap_idx.size != 1:
        return None

    start_perim_loc_of_flood_entry = region_boundary_locs[max_gap_idx[0]]

    # see if subsequent perimeter locations were also part of the contour,
    # adding one to the last one we found
    subsequent_region_border_locs = region_boundary_locs[
        region_boundary_locs > start_perim_loc_of_flood_entry
    ]

    flood_fill_entry_point = start_perim_loc_of_flood_entry

    for loc in subsequent_region_border_locs:
        if loc == flood_fill_entry_point + 1:
            flood_fill_entry_point = loc

    # we should hit the first interior empty point of the contour
    # by moving forward one pixel around the perimeter
    flood_fill_entry_point += 1

    # now to convert our perimeter location back to an image coordinate
    if flood_fill_entry_point < img_w:
        flood_fill_entry_coords = (flood_fill_entry_point, 0)
    elif flood_fill_entry_point < img_w + img_h:
        flood_fill_entry_coords = (img_w - 1, flood_fill_entry_point - img_w + 1)
    elif flood_fill_entry_point < img_w * 2 + img_h:
        flood_fill_entry_coords = (img_h + (2 * img_w) - 3 - flood_fill_entry_point, img_h - 1)
    else:
        flood_fill_entry_coords = (0, perimeter - flood_fill_entry_point)

    flood_fill_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(mask, flood_fill_mask, tuple(flood_fill_entry_coords), 255)

    return mask


def find_border_by_mask(
        signal_mask,
        contour,
        signal_threshold=0.7,
        max_dilate_percentage=2.0,
        spread=9,
        dilate_iterations=1,
        plot=False,
        figsize=(16, 4)
):
    contour_mask = np.zeros(signal_mask.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, cv2.FILLED)

    area = np.sum(contour_mask > 0)
    max_dilation_area = area + int(max_dilate_percentage * area)

    if max_dilation_area > contour_mask.size:
        max_dilation_area = contour_mask.size

    # create a baseline from the original border
    filled_c_mask_erode = cv2.erode(
        contour_mask,
        circle_strel,
        iterations=dilate_iterations
    )
    border_mask = contour_mask - filled_c_mask_erode
    border_signal_mask = np.bitwise_and(border_mask, signal_mask)

    max_signal = np.sum(border_signal_mask > 0) / np.sum(border_mask > 0)

    filled_c_mask_dilate = contour_mask.copy()
    signal_profile = [max_signal]
    count = 0

    while area < max_dilation_area:
        count += 1

        last_mask = filled_c_mask_dilate.copy()
        filled_c_mask_dilate = cv2.dilate(
            last_mask,
            circle_strel,
            iterations=dilate_iterations
        )
        area = np.sum(filled_c_mask_dilate > 0)
        border_mask = filled_c_mask_dilate - last_mask
        border_signal_mask = np.bitwise_and(border_mask, signal_mask)

        if border_mask.max() == 0:
            new_signal = 0
        else:
            new_signal = np.sum(border_signal_mask > 0) / np.sum(border_mask > 0)

        signal_profile.append(new_signal)

        if max_signal > 0 and new_signal == 0 and count > 5:
            break

        if new_signal >= max_signal:
            max_signal = new_signal

    final_dilate_iter = 0

    if max_signal > signal_threshold:

        n = len(signal_profile)
        x = list(range(n))
        min_a = 0
        max_a = 1
        min_sigma = 0
        max_sigma = n / 2
        max_idx = np.argmax(signal_profile)

        p0 = [1, max_idx, 1]
        min_mu = max_idx - spread
        max_mu = max_idx + spread

        if max_idx - spread < 0:
            low_bound = 0
        else:
            low_bound = max_idx - spread

        if max_idx + spread > n:
            high_bound = n
        else:
            high_bound = max_idx + spread

        iso_x = x[low_bound:high_bound]
        iso_data = signal_profile[low_bound:high_bound]

        popt, pcov = optimize.curve_fit(
            gauss,
            iso_x,
            iso_data,
            p0=p0,
            maxfev=50000,
            bounds=(
                (min_a, min_mu, min_sigma),
                (max_a, max_mu, max_sigma)
            )
        )

        final_dilate_iter = int(np.round(popt[1] + 1.50 * popt[2]))

        if plot:
            print(popt)
            plt.figure(figsize=figsize)
            plt.bar(x, signal_profile)
            plt.plot(
                x,
                gauss(np.array(x), *popt),
                color='lime',
                linestyle='--',
                linewidth=3
            )
            plt.axvline(final_dilate_iter, color='k', linewidth=5)

            plt.show()

    if final_dilate_iter > 0:
        final_mask = cv2.dilate(
            contour_mask,
            circle_strel,
            iterations=final_dilate_iter
        )
        orig = False
    else:
        final_mask = contour_mask
        orig = True

    return final_mask, max_signal, orig


def find_contour_union(contour_list, img_shape):
    union_mask = np.zeros(img_shape, dtype=np.uint8)

    for c in contour_list:
        c_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(c_mask, [c], 0, 255, cv2.FILLED)
        union_mask = cv2.bitwise_or(union_mask, c_mask)

    return union_mask


def generate_background_contours(
        hsv_img,
        non_bg_contours,
        n_segments=200,
        remove_border_contours=True,
        plot=False
):
    img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    non_bg_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.drawContours(non_bg_mask, non_bg_contours, -1, 255, cv2.FILLED)

    bg_mask_img = cv2.bitwise_and(img, img, mask=~non_bg_mask)

    segments = slic(
        bg_mask_img,
        n_segments=n_segments,
        compactness=100,
        sigma=1,
        enforce_connectivity=True
    )

    masked_segments = cv2.bitwise_and(segments, segments, mask=~non_bg_mask)

    all_contours = []

    for label in np.unique(masked_segments):
        if label == 0:
            continue

        mask = masked_segments == label
        mask.dtype = np.uint8
        mask[mask == 1] = 255
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=3)

        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )

        all_contours.extend(contours)

    bkgd_contour_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.drawContours(bkgd_contour_mask, all_contours, -1, 255, -1)

    all_contours = filter_contours_by_size(
        bkgd_contour_mask, min_size=64 * 64,
        max_size=1000 * 1000
    )

    if remove_border_contours:
        border_contours, all_contours = find_border_contours(
            all_contours,
            img.shape[0],
            img.shape[1]
        )

    if plot:
        bkgd_contour_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.drawContours(bkgd_contour_mask, all_contours, -1, 255, -1)

        plt.figure(figsize=(16, 16))
        plt.imshow(cv2.cvtColor(bkgd_contour_mask, cv2.COLOR_GRAY2RGB))

        bg_mask_img = cv2.bitwise_and(img, img, mask=bkgd_contour_mask)
        plt.figure(figsize=(16, 16))
        plt.imshow(bg_mask_img)

        plt.show()

    return all_contours


def elongate_contour(contour, img_shape, extend_length):
    c_mask = np.zeros(img_shape, dtype=np.uint8)

    cv2.drawContours(c_mask, [contour], -1, 255, -1)

    rect = cv2.minAreaRect(contour)

    cx, cy = rect[0]
    w, h = rect[1]
    angle = rect[2]

    if w <= 1 or h <= 1 or extend_length < 0:
        return contour

    if isinstance(extend_length, float) and extend_length <= 1.0:
        extend_length = int(extend_length * max(w, h)) + 1

    if w > h:
        angle = angle - 90

    mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    c_mask_rot = cv2.warpAffine(c_mask, mat, img_shape)
    c_mask_rot[c_mask_rot > 0] = 255

    y_locs = np.where(c_mask_rot > 0)[0]
    y_min = y_locs.min()
    y_max = y_locs.max()
    y_mid = int(np.round(np.average([y_min, y_max])))

    top_x_locs = np.where(c_mask_rot[y_min + 1, :] > 0)[0]
    mid_x_locs = np.where(c_mask_rot[y_mid, :] > 0)[0]
    bottom_x_locs = np.where(c_mask_rot[y_max - 1, :] > 0)[0]

    mid_x_min = mid_x_locs.min()
    mid_x_max = mid_x_locs.max()
    mid_x_mid = int(np.round(np.average([mid_x_min, mid_x_max])))
    mid_width = mid_x_max - mid_x_min

    if len(top_x_locs) > 0:
        top_x_min = top_x_locs.min()
        top_x_max = top_x_locs.max()
        top_x_mid = int(np.round(np.average([top_x_min, top_x_max])))
        extend_top = True
    else:
        extend_top = False

    if len(bottom_x_locs) > 0:
        bottom_x_min = bottom_x_locs.min()
        bottom_x_max = bottom_x_locs.max()
        bottom_x_mid = int(np.round(np.average([bottom_x_min, bottom_x_max])))
        extend_bottom = True
    else:
        extend_bottom = False

    mid_coord = (mid_x_mid, y_mid)
    new_c_mask_rot = c_mask_rot.copy()

    if extend_top:
        top_coord = (top_x_mid, y_min)

        top_angle = np.math.atan2(top_coord[1] - mid_coord[1], top_coord[0] - mid_coord[0])
        top_angle = top_angle * 180 / np.pi

        cv2.ellipse(
            new_c_mask_rot,
            top_coord,
            (extend_length, int(mid_width / 4)),
            top_angle,
            0,
            360,
            255,
            -1
        )

    if extend_bottom:
        bottom_coord = (bottom_x_mid, y_max)

        bottom_angle = np.math.atan2(bottom_coord[1] - mid_coord[1],
                                     bottom_coord[0] - mid_coord[0])
        bottom_angle = bottom_angle * 180 / np.pi

        cv2.ellipse(
            new_c_mask_rot,
            bottom_coord,
            (extend_length, int(mid_width / 4)),
            bottom_angle,
            0,
            360,
            255,
            -1
        )

    inv_mat = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    c_mask_new = cv2.warpAffine(new_c_mask_rot, inv_mat, img_shape)

    # fix interpolation artifacts
    c_mask_new[c_mask_new > 0] = 255

    contours, hierarchy = cv2.findContours(
        c_mask_new.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours[0]


def gauss(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2*sigma**2))


# from scipy-cookbook:
# https://scipy-cookbook.readthedocs.io/items/FittingData.html
def gaussian_2d(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height * np.exp(
        -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2
    )


def _moments_gaussian_2d(data):
    """
    Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments
    """
    total = data.sum()
    x, y = np.indices(data.shape)
    x = (x * data).sum() / total
    y = (y * data).sum() / total

    col = data[:, int(y)]
    width_x = np.sqrt(
        np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum()
    )

    row = data[int(x), :]
    width_y = np.sqrt(
        np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum()
    )

    height = data.max()

    return height, x, y, width_x, width_y


def _error_function_gaussian_2d(data):
    return lambda p: np.ravel(
        gaussian_2d(*p)(*np.indices(data.shape)) - data
    )


def fit_gaussian_2d(data):
    """
    Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit
    """
    params = np.array(_moments_gaussian_2d(data))
    lsq_results = optimize.leastsq(_error_function_gaussian_2d(data), params)

    return lsq_results[0]


def calculate_nonuniform_field(single_channel):
    """
    Calculates the non-uniformity of a 2-D array, most typically used to
    correct for uneven "lightness" of an image.
    :param single_channel: Single channel 2-D NumPy array
    :return: 2-D NumPy array (float64)
    """
    # "Elevate" any black pixel values to half the median to better fit
    # the data. Black pixels provide no details as to the variation of
    # luminance within an image
    med = np.median(single_channel[single_channel > 0])

    tmp_img = single_channel.copy()
    tmp_img[tmp_img <= med / 2.0] = int(med / 2.0)

    params = fit_gaussian_2d(tmp_img)
    fit = gaussian_2d(*params)

    non_uni_field = fit(*np.indices(tmp_img.shape))

    return non_uni_field


def correct_nonuniformity(single_channel, mask=None):
    if mask is not None:
        masked_img = cv2.bitwise_and(single_channel, single_channel, mask=mask)
    else:
        masked_img = single_channel.copy()

    non_uni_field = calculate_nonuniform_field(masked_img)

    non_uni_field = non_uni_field.round().astype(np.uint8)

    field_corr = ~non_uni_field
    field_corr -= field_corr.min()

    single_channel_corr = single_channel.copy().astype(np.uint16)
    single_channel_corr += field_corr
    single_channel_corr[single_channel_corr > 255] = 255
    single_channel_corr = single_channel_corr.astype(np.uint8)

    return single_channel_corr
