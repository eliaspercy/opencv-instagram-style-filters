import cv2
import numpy as np
from numpy import exp, sqrt
from math import floor, ceil
import sys
import random
from typing import Tuple, Union, Dict, List
from matplotlib import pyplot as plt
import argparse

# ------------------------------------------------------- INITIALISATION --------------------------------------------- #

FILTERS = {'light-leak', 'problem1', 'pencil', 'problem2', 'beautify', 'problem3', 'swirl', 'problem4'}
INPUT_FILTERS = []
try:
    IMG_PATH = sys.argv[1]
except IndexError:
    IMG_PATH = 'face1.jpg'

ig = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)

# loop for filters
for argument in range(2, len(sys.argv)):
    try:
        f = str(sys.argv[argument]).lower()
        if f in FILTERS:
            INPUT_FILTERS.append(f)
    except IndexError:
        break

# Collect arguments
parser = argparse.ArgumentParser()

# Light leak args
parser.add_argument('-rb', '--rainbow', dest='rainbow', action='store_true', default=False,
                    help='Turn rainbow mode on')
parser.add_argument('-rbs', '--rainbow-strength', dest='rainbow_strength', default=0.4, type=float,
                    help='Float: determine the relative strength of the rainbow')
parser.add_argument('-blc', '--blending-coefficient', dest='blending_coefficient', default=0.6, type=float,
                    help='Float: determine the blending between the image and the light.')
parser.add_argument('-dk', '--darkening-coefficient', dest='darkening_coefficient', default=0.3, type=float,
                    help='Float: determine how dark the image becomes beside the light.')
parser.add_argument('-brc', '--brightening-coefficient', dest='brightening_coefficient', default=0.5, type=float,
                    help='Float: determine the the maximum brightness of the light leak.')
parser.add_argument('-pos', '--light-leak-position', dest='position', default=0.5, type=float,
                    help='Float: determine the relative position of the light leak.')

# Pencil args
parser.add_argument('-co', '--pencil-colour', dest='colour', default='none', type=str,
                    help='Str: determine the colour of the pencils used for the sketch')
parser.add_argument('-csr', '--canvas-stroke-rate', dest='stroke_rate', default=0.4, type=float,
                    help='Float: determine the rate of strokes on the canvas.')
parser.add_argument('-csw', '--canvas-stroke-width', dest='stroke_width', default=0.23, type=float,
                    help='Float: determine the width of the strokes on the canvas (i.e., amount of motion blur).')
parser.add_argument('-cbc', '--canvas-blending-coefficient', dest='blend', default=0.3, type=float,
                    help='Float: determine the amount by which the canvas and the image are blended.')
parser.add_argument('-sap', '--salt-and-pepper', dest='salt_and_pepper', action='store_true', default=False,
                    help='Decide whether or not to use salt and pepper noise when generating the canvas.')

# Beautifying args
parser.add_argument('-sc', '--bilateral-smoothing-coefficient', dest='smoothing_coefficient', default=2, type=int,
                    help='Int: Determines the strength of the bilateral filter applied to the image.')
parser.add_argument('-sf', '--sub-filter', dest='sub_filter', default='mid-autumn', type=str,
                    help='Str: Decides which sub-filter to use. Options: ...')
parser.add_argument('-ci', '--contrast-intensity', dest='contrast_intensity', default=40, type=int,
                    help='Int: determines the intensity of the contrast when the late-summer filter is applied.')
parser.add_argument('-hst', '--display-histogram', dest='histogram', action='store_true', default=False,
                    help='Toggle this to display a histogram before and after the image filter is applied.')

# Swirl args
parser.add_argument('-c', '--swirl-centre', dest='centre', default=[100, 100], nargs='+', type=int,
                    help='(Int, Int): The coordinates of the center of the swirl.')
parser.add_argument('-r', '--swirl-radius', dest='radius', default=int(160), type=int,
                    help='Int: The radius of the swirl.')
parser.add_argument('-s', '--swirl-strength', dest='strength', default=-2.5, type=float,
                    help='Float: The strength of the swirl. Can be positive (clockwise) or negative (anti-clockwise).')
parser.add_argument('-nn', '--nearest-neighbour-interpolation', dest='bilinear', action='store_false', default=True,
                    help='If set, the swirl pixels are interpolated via nearest neighbour interpolation; '
                         'by default, however, bilinear interpolation is used.')
parser.add_argument('-re', '--reverse-swirl', dest='reverse', action='store_true', default=False,
                    help='Decide whether to inverse the swirl and undertake image subtracting to compare the original'
                         'image with the reversed swirl image. For debugging purposes.')
parser.add_argument('-sub', '--subtract', dest='subtract', action='store_true', default=False,
                    help='Use this argument to enforce image subtraction occurs between the original image and the '
                         'image after the swirl filter is applied (generally to be used in conjunction with -re.')
parser.add_argument('-lpf', '--low-pas-filter', dest='butterworth', action='store_true', default=False,
                    help='Enabling this argument will force the image to undergo a butterworth low-pass filter prior to'
                         'the swirl transformation. By default, the filter won\'t be applied.')
parser.add_argument('-cof', '--cut-off-frequency', dest='cut_off_frequency', default=70, type=int,
                    help='Int: The cut off distance for the Butterworth low-pass filter.')
parser.add_argument('-lpo', '--low-pass-order', dest='order', default=1, type=float,
                    help='Float: The order for the Butterworth low-pass filter.')

# Output argument
parser.add_argument('-many', '--multiple-filters', dest='multiple_filters', action='store_true', default=False,
                    help='Toggle this if you intend to use multiple filters on the same image: this will entail only'
                         'outputting the image after all filters have been applied.')
parser.add_argument('-o', '--out-file', dest='out_file', default='n/a', type=str,
                    help='Specify a file to output the image to; if flag not included, image is simply displayed.')

args, unknown = parser.parse_known_args()


# ------------------------------------------- PART 1: LIGHT LEAK ----------------------------------------------------- #

def adjust_brightness(pix: np.ndarray, alpha: float, beta: float = 0) -> np.ndarray:
    # Alpha: Contrast control; Beta: Brightness control
    return np.clip(alpha * pix + beta, 0, 255)  # np.clip used to ensure pixel values do not exceed 255


def fade_and_blend(img: np.ndarray, mask: np.ndarray,
                   left_fade: int, right_fade: int, alpha: float, beta: float) -> np.ndarray:
    """
    Function requires that the two images have the same size
    """
    rows, cols = img.shape[0], img.shape[1]
    inc1 = (1 - alpha) / left_fade
    inc2 = beta / left_fade
    alpha_var = 1
    beta_var = 0
    for j in range(0, left_fade):
        try: img[:, j] = np.clip(alpha_var * img[:, j] + beta_var * mask[:, j], 0, 255)
        except IndexError: pass
        alpha_var -= inc1
        beta_var += inc2
    try:
        img[:, left_fade:right_fade] = \
            np.clip(alpha_var * img[:, left_fade:right_fade] + beta_var * mask[:, left_fade:right_fade], 0, 255)
    except IndexError: pass
    for j in range(right_fade, cols):
        alpha_var += inc1
        beta_var -= inc2
        try: img[:, j] = np.clip(alpha_var * img[:, j] + beta_var * mask[:, j], 0, 255)
        except IndexError: pass
    return img


def apply_rainbow_mask(img: np.ndarray, start: int, fade_left: int,
                       fade_right: int, end: int, strength: float) -> np.ndarray:
    """
    [B, G, R] Rainbow:
    Red: [0, 0, 255]
    Orange: [0, 165, 255]
    Yellow: [0, 255, 255]
    Green: [0, 255, 0]
    Blue: [255, 0, 0]
    Indigo: [75, 0, 130]
    Violet: [148, 0, 211]
    """
    rows, cols = img.shape[0], img.shape[1]
    width = end - start

    # 1. Initialise the rainbow
    rainbow = np.zeros((rows, width, 3), dtype=np.uint8)

    # 1.1 Get widths and colour indices
    colour_width = width // 7
    red_start = colour_width
    orange_start = red_start + colour_width
    yellow_start = orange_start + colour_width
    green_start = yellow_start + colour_width
    blue_start = green_start + colour_width
    indigo_start = blue_start + colour_width
    violet_start = indigo_start + colour_width

    # 1.2 Get increments
    _0_to_165 = 165 / colour_width
    _0_to_255 = 255 / colour_width
    _0_to_130 = 130 / colour_width
    _75_to_255 = 180 / colour_width
    _75_to_148 = 73 / colour_width
    _130_to_211 = 81 / colour_width
    _165_to_255 = 90 / colour_width

    # 2. Create the rainbow
    blue_guy, green_guy, red_guy = 0, 0, 255
    try: rainbow[:, 0:red_start] = blue_guy, green_guy, red_guy
    except IndexError: pass
    for j in range(red_start, orange_start):
        try: rainbow[:, j] = blue_guy, green_guy, red_guy
        except IndexError: pass
        green_guy += _0_to_165
    for j in range(orange_start, yellow_start):
        try: rainbow[:, j] = blue_guy, green_guy, red_guy
        except IndexError: pass
        green_guy += _165_to_255
    for j in range(yellow_start, green_start):
        try: rainbow[:, j] = blue_guy, green_guy, red_guy
        except IndexError: pass
        red_guy -= _0_to_255
    for j in range(green_start, blue_start):
        try: rainbow[:, j] = blue_guy, green_guy, red_guy
        except IndexError: pass
        blue_guy += _0_to_255
        green_guy -= _0_to_255
    for j in range(blue_start, indigo_start):
        try: rainbow[:, j] = blue_guy, green_guy, red_guy
        except IndexError: pass
        blue_guy -= _75_to_255
        red_guy += _0_to_130
    for j in range(indigo_start, violet_start):
        try: rainbow[:, j] = blue_guy, green_guy, red_guy
        except IndexError: pass
        blue_guy += _75_to_148
        red_guy += _130_to_211
    try: rainbow[:, violet_start:width] = blue_guy, green_guy, red_guy
    except IndexError: pass

    # 3. Now blend the rainbow with the input image (i.e., apply the rainbow mask)
    img[:rows, start:end] = fade_and_blend(img[:rows, start:end], rainbow, fade_left - start,
                                           fade_right - start, 1 - strength, strength)

    # 4. Return the image now with the rainbow applied
    return img


def light_leak_mask(img: np.ndarray, blending_range: int, width: int, darkness: float,
                    brightness: float, centre: float, rainbow: bool, rainbow_strength: float) -> np.ndarray:
    """
    outline:
    dark range        - adjustable parameter (darkness)
    brighten range    - adjustable parameter (blending)
    bright range      - [optional] adjustable parameter (brightness)
    darken range      - adjustable parameter (blending)
    dark range        - adjustable parameter (darkness)
    """
    rows, cols = img.shape[0], img.shape[1]
    ll_img = np.zeros_like(img)
    start_of_light_leak = int(centre * cols) - int(0.5 * width) - blending_range
    try:
        alpha_inc = (brightness - darkness) / blending_range
        beta_inc = (10 * brightness) / blending_range
    except ZeroDivisionError:
        alpha_inc = beta_inc = 0

    brighten_start = start_of_light_leak
    brighten_end = brighten_start + blending_range
    darken_start = brighten_end + width
    darken_end = darken_start + blending_range
    alpha_var = darkness
    beta_var = 0

    # Apply light-leak filter
    ll_img[:, 0:brighten_start] = adjust_brightness(img[:, 0:brighten_start], darkness)
    for i in range(brighten_start, brighten_end):
        alpha_var += alpha_inc
        beta_var += beta_inc
        try: ll_img[:, i:i + 1] = adjust_brightness(img[:, i:i + 1], alpha_var, beta_var)
        except IndexError: pass
    try: ll_img[:, brighten_end:darken_start] = adjust_brightness(img[:, brighten_end:darken_start],
                                                                  alpha_var, beta_var)
    except IndexError: pass
    for i in range(darken_start, darken_end):
        try: ll_img[:, i:i + 1] = adjust_brightness(img[:, i:i + 1], alpha_var, beta_var)
        except IndexError: pass
        alpha_var -= alpha_inc
        beta_var -= beta_inc
    try: ll_img[:, darken_end:cols] = adjust_brightness(img[:, darken_end:cols], darkness)
    except IndexError: pass

    # If the rainbow filter was selected, apply the rainbow mask
    if rainbow: ll_img = apply_rainbow_mask(ll_img, brighten_start, brighten_end,
                                            darken_start, darken_end, rainbow_strength)

    # Return the now altered image
    return ll_img


def problem1(image_path=IMG_PATH, enforce_img=False, enforced_img=ig, rainbow=args.rainbow,
             rainbow_strength=args.rainbow_strength, filter_centre=args.position, filter_width=0.06,
             darkening_coefficient=args.darkening_coefficient, brightening_coefficient=args.brightening_coefficient,
             blending_coefficient=args.blending_coefficient) -> Union[np.ndarray, None]:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) if not enforce_img else enforced_img
    if image is None:
        print("Error: Image not found.")
        return
        # sys.exit(1)
    rows, cols = image.shape[0], image.shape[1]
    rainbow_strength = float(np.clip(rainbow_strength, 0, 1))
    filter_centre = float(np.clip(filter_centre, 0.1, 1))
    darkening_coefficient = int(np.clip(10 * darkening_coefficient, 0, 9))
    brightening_coefficient = int(np.clip(10 * brightening_coefficient, 0, 9))
    blending_coefficient = int(np.clip(10 * blending_coefficient, 0, 9))
    if rainbow: blending_coefficient += 2

    DARKNESS_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    BLENDING_VALUES = [0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.075, 0.1, 0.2, 0.3, 0.35, 0.4]
    BRIGHTNESS_VALUES = [1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]

    try:
        darkness_value = DARKNESS_VALUES[darkening_coefficient]
        blending_value = BLENDING_VALUES[blending_coefficient]
        brightness_value = BRIGHTNESS_VALUES[brightening_coefficient]
    except IndexError:
        print("Invalid parameters.")
        sys.exit()

    light_leak_img = light_leak_mask(image, int(blending_value * cols), int(filter_width * cols), darkness_value,
                                     brightness_value, filter_centre, rainbow, rainbow_strength)
    if not args.multiple_filters:
        cv2.imshow('Input image', image)
        cv2.imshow('Light-leak filter', light_leak_img)
        key = cv2.waitKey(0)
        if key == ord('x'):
            cv2.destroyAllWindows()
    return light_leak_img


# ----------------------------------------- PART 2: Pencil/Charcoal -------------------------------------------------- #

def create_salt_and_pepper_mask(img: np.ndarray, noise_prob: float) -> np.ndarray:
    rows, cols = img.shape[0], img.shape[1]
    noise = np.full((rows, cols), 255, dtype=np.uint8)
    noise_prob = np.clip(noise_prob, 0, 1)
    noise_prob = 1 - noise_prob
    # Create salt and pepper noise
    for i in range(rows):
        for j in range(cols):
            rand = random.uniform(0, 1)
            if rand < 0.7*noise_prob: noise[i, j] = 255
            # elif rand > (1 - (noise_prob / 2)): noise[i, j] = 255
            else: noise[i, j] = 0
    return noise


def create_gaussian_noise_mask(img: np.ndarray, stroke_quantity: float) -> np.ndarray:
    rows, cols = img.shape[0], img.shape[1]
    if stroke_quantity == 0: return np.full((rows, cols), 255, np.uint8)
    stroke_quantity = float(np.clip(stroke_quantity, 0, 1))
    stroke_quantity = 1 - stroke_quantity
    sigma = 10 * stroke_quantity
    gauss = np.random.normal(0, sigma ** 0.5, (rows, cols))
    gauss = gauss.reshape(rows, cols).astype(np.uint8)
    return gauss


def apply_motion_blur(img: np.ndarray, kernel_size: int, direction: str) -> np.ndarray:
    """
    1. Initialise the blur kernel with the inputted kernel_size
    2. Fill up the middle row or column of the blur kernel with ones
    3. Normalise the blur kernel by via division with its size and return the input image with convolution applied
    """
    motion_blur_kernel = np.zeros((kernel_size, kernel_size))
    middle = int(0.5 * (kernel_size - 1))
    if 'h' in direction: motion_blur_kernel[middle, :] = np.ones(kernel_size)
    if 'v' in direction: motion_blur_kernel[:, middle] = np.ones(kernel_size)
    motion_blur_kernel /= kernel_size
    return cv2.filter2D(img, -1, motion_blur_kernel)  # We're allowed to use filter2D


def construct_canvas(img: np.ndarray, stroke_rate: float, stroke_width: int, sap, direction: str) -> np.ndarray:
    noise = create_salt_and_pepper_mask(img, stroke_rate) if sap else create_gaussian_noise_mask(img, stroke_rate)
    canvas = apply_motion_blur(noise, stroke_width, direction)
    return np.uint8(canvas)


def apply_canvas(img: np.ndarray, stroke_rate: float, stroke_width: int,
                 blend: float, sap: bool, direction: str) -> np.ndarray:
    blend = float(np.clip(blend, 0, 1))
    canvas = construct_canvas(img, stroke_rate, stroke_width, sap, direction)
    img = np.clip((max(1 - blend, 0.9)) * img + blend * canvas - blend * 200, 0, 255)
    return np.uint8(img)


def apply_mean_filter(img: np.ndarray, kernel_size: int) -> np.ndarray:
    mean_kernel = np.ones((kernel_size, kernel_size))
    mean_kernel /= kernel_size ** 2
    return cv2.filter2D(img, -1, mean_kernel)


def apply_pencil_effect(img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = np.clip(apply_mean_filter(img, 4), 1, 255)
    img = np.clip(img * 255.0 / img_blur, 0, 255)
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def sketch(img: np.ndarray, colour: str, blend: float, stroke_rate: float, stroke_width: int, sap: bool) -> np.ndarray:
    rows, cols = img.shape[0], img.shape[1]
    img = apply_pencil_effect(img)
    if colour.lower() == 'none':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = apply_canvas(img, stroke_rate, stroke_width, blend, sap, 'hv')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif colour.lower() == 'green':
        # apply noise to blue and red channel
        blue, green, red = cv2.split(img)
        blue = apply_canvas(blue, stroke_rate/2, stroke_width, blend, not sap, 'h')
        red = apply_canvas(red, stroke_rate, stroke_width, blend, sap, 'hv')
        green = np.full((rows, cols), 255, np.uint8)
        img = cv2.merge((blue, green, red))
    elif colour.lower() == 'red':
        blue, green, red = cv2.split(img)
        green = apply_canvas(green, stroke_rate, stroke_width, blend, not sap, 'vh')
        blue = apply_canvas(blue, stroke_rate/2, stroke_width, blend, sap, 'vh')
        red = np.full((rows, cols), 255, np.uint8)
        img = cv2.merge((blue, green, red))
    elif colour.lower() == 'purple':
        blue, green, red = cv2.split(img)
        blue = np.full((rows, cols), 255, np.uint8)
        green = apply_canvas(green, stroke_rate, stroke_width, blend, sap, 'hv')
        red = apply_canvas(blue, stroke_rate/2, stroke_width, blend, not sap, 'v')
        img = cv2.merge((blue, green, red))
    elif colour.lower() == 'blue':
        blue, green, red = cv2.split(img)
        blue = np.full((rows, cols), 255, np.uint8)
        green = apply_canvas(green, stroke_rate, stroke_width, blend, not sap, 'hv')
        red = apply_canvas(blue, stroke_rate/2, stroke_width, blend, sap, 'v')
        img = cv2.merge((blue, green, red))
    return img


def problem2(image_path=IMG_PATH, enforce_img=False, enforced_img=ig, colour=args.colour,
             canvas_stroke_rate=args.stroke_rate, canvas_blending_coefficient=args.blend,
             canvas_stroke_width=args.stroke_width, salt_and_pepper_mask=args.salt_and_pepper
             ) -> Union[np.ndarray, None]:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) if not enforce_img else enforced_img
    if image is None:
        print("Error: Image not found.")
        return
        # sys.exit(1)
    colours = {'purple', 'blue', 'green', 'red'}
    if colour.lower() == 'none':
        pass
    elif colour.lower() not in colours:
        print("Unavailable colour, using none")
        colour = 'none'
    canvas_stroke_width = int(100 * canvas_stroke_width)
    sketch_img = sketch(image, colour, canvas_blending_coefficient, canvas_stroke_rate,
                        canvas_stroke_width, salt_and_pepper_mask)
    if not args.multiple_filters:
        cv2.imshow('Input image', image)
        cv2.imshow('Sketch filter', sketch_img)
        key = cv2.waitKey(0)  # wait
        if key == ord('x'):
            cv2.destroyAllWindows()
    return sketch_img


# --------------------------------------------------- PART 3: Beautify ----------------------------------------------- #

def gaussian_function(x: Union[float, np.ndarray], sigma: float) -> Union[float, np.ndarray]:
    return exp(-(x ** 2) / (2 * (sigma ** 2)))  # using np.exp because it works with arrays


def apply_padding(img: np.ndarray, pad: int) -> np.ndarray:  # O(n) for constructing the new matrix, rest is O(1)
    """
    :param img: np.ndarray representing an image
    :param pad: The number of pixels to pad the image by
    :return: The input image, as an np.ndarray, with padding of size pad
    """
    rows, cols = img.shape[0], img.shape[1]
    try: c = img.shape[2]
    except IndexError: c = 1
    padded = np.zeros((rows + 2 * pad, cols + 2 * pad, c), dtype=np.int32)

    # Original image in the middle
    padded[pad:rows + pad, pad:cols + pad] = img

    # Sides
    padded[0:pad, pad:cols + pad] = img[0:pad, 0:rows]  # Top
    padded[rows + pad: rows + 2 * pad, pad:cols + pad] = img[rows - pad:rows, 0:cols]  # Bottom
    padded[pad:rows + pad, 0:pad] = img[0:rows, 0:pad]  # Left
    padded[pad:rows + pad, cols + pad:cols + 2 * pad] = img[0:rows, cols - pad:cols]  # Right

    # Corners
    padded[0:pad, 0:pad] = img[0:pad, 0:pad]  # Top left
    padded[0:pad, cols + pad:cols + 2 * pad] = img[0:pad, cols - pad:cols]  # Top right
    padded[rows + pad:rows + 2 * pad, 0:pad] = img[rows - pad:rows, 0:pad]  # Bottom left
    padded[rows + pad:rows + 2 * pad, cols + pad:cols + 2 * pad] = img[rows - pad:rows, cols - pad:cols]  # Bottom right

    return padded.astype(np.float64)


def construct_bilateral_distance_matrix(kernel_size: int, sigma: float) -> np.ndarray:
    dist_matrix = np.zeros((2 * kernel_size + 1, 2 * kernel_size + 1, 3), dtype=np.float64)
    for i in range(-kernel_size, kernel_size + 1):
        for j in range(-kernel_size, kernel_size + 1):
            if i == j == 0:
                dist_matrix[i + kernel_size, j + kernel_size] = 1.0
                continue
            dist = sqrt(i ** 2 + j ** 2)
            dist_matrix[i + kernel_size, j + kernel_size] = gaussian_function(dist, sigma)
    return dist_matrix


def apply_bilateral_filter(img: np.ndarray, kernel_size: int = 1,
                           sig_space: float = 2.0, sig_sim: float = 15.0) -> np.ndarray:
    """
    :param img: np.ndarray representing the image to be filtered
    :param kernel_size: The size radius of the nxn neighbourhood to be examined, where n = 2*kernel_size + 1
    :param sig_space: Sigma value for spatial proximity/distance
    :param sig_sim: Sigma value for image pixel similarity
    :return: The input image with the bilateral filter applied
    """
    if kernel_size == 0: return img
    rows, cols = img.shape[0], img.shape[1]
    pad = kernel_size
    padded_img = apply_padding(img, pad)
    new_img = np.zeros(img.shape, dtype=np.float64)

    # Construct the distance matrix for the filter, according to the kernel size
    dist_matrix = construct_bilateral_distance_matrix(kernel_size, sig_space)

    for i in range(kernel_size, rows + kernel_size):  # O(n^2), two for loops
        for j in range(kernel_size, cols + kernel_size):
            window = padded_img[i - kernel_size:i + kernel_size + 1, j - kernel_size: j + kernel_size + 1]
            filter_matrix = gaussian_function(window - padded_img[i, j], sig_sim)
            filter_matrix = np.multiply(filter_matrix, dist_matrix)
            filter_matrix /= np.sum(filter_matrix, keepdims=False, axis=(0, 1))  # Normalise
            filtered = np.sum(np.multiply(filter_matrix, window), axis=(0, 1))
            new_img[i - pad, j - pad] = np.clip(filtered, 0, 255)

    return np.uint8(new_img)


def apply_look_up_table(img: np.ndarray, luts: Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]) -> np.ndarray:
    rows, cols = img.shape[0], img.shape[1]
    blut, glut, rlut = luts
    blue, green, red = cv2.split(img)
    for i in range(rows):
        for j in range(cols):
            blue[i, j] = blut[blue[i, j]]
            green[i, j] = glut[green[i, j]]
            red[i, j] = rlut[red[i, j]]
    return cv2.merge((blue, green, red))


def create_look_up_table(a: List[Tuple[int, int]]) -> Dict[int, int]:
    if a[0][0] != 0: raise ValueError
    if a[-1][0] != 255: raise ValueError
    if a[0][1] < 0: raise ValueError
    if a[-1][1] > 255: raise ValueError
    n: int = len(a)
    a3: List[int] = [a[i][1] - a[i][0] for i in range(n)]
    incs: List[float] = [(a3[i + 1] - a3[i]) / (a[i + 1][0] - a[i][0]) for i in range(n - 1)] + [1]
    curve, inc = a3[0], incs[0]
    lut: Dict[int, int] = {}
    j: int = 0
    for i in range(256):
        if i == a[j][0]:
            inc = incs[j]
            j += 1
        lut[i] = int(np.clip(round(i + curve), 0, 255))
        curve += inc
    return lut


def get_summer_luts(contrast_intensity: int):
    intensity = int(np.clip(contrast_intensity, 0, 64))
    lut = create_look_up_table([(0, 0), (64, 64 - intensity), (127, 127), (191, 191 + intensity), (255, 255)])
    return lut, lut, lut


def get_autumn_luts():
    blut = create_look_up_table([(0, 35), (40, 75), (82, 126), (120, 140), (175, 190), (220, 215), (255, 230)])
    glut = create_look_up_table([(0, 0), (38, 70), (85, 120), (125, 160), (172, 186), (218, 210), (255, 230)])
    rlut = create_look_up_table([(0, 30), (30, 70), (130, 192), (170, 200), (233, 233), (255, 245)])
    return blut, glut, rlut


def get_spring_luts():
    blut = create_look_up_table([(0, 25), (40, 60), (90, 100), (120, 130), (164, 170), (212, 195), (255, 210)])
    glut = create_look_up_table([(0, 30), (40, 55), (90, 115), (135, 170), (170, 195), (215, 218), (255, 230)])
    rlut = create_look_up_table([(0, 25), (45, 80), (85, 135), (120, 180), (230, 240), (255, 255)])
    return blut, glut, rlut


def apply_filter_main(img: np.ndarray, style: str, ci: int) -> np.ndarray:
    null_lut = {i: i for i in range(256)}
    blut, glut, rlut = null_lut, null_lut, null_lut

    if style.lower() == 'none':
        pass
    elif style.lower() == 'late-summer':
        # mid-autumn
        # high contrast
        img = apply_look_up_table(img, get_autumn_luts())
        img = apply_look_up_table(img, get_summer_luts(ci))
    elif style.lower() == 'mid-autumn':
        # mid-autumn
        img = apply_look_up_table(img, get_autumn_luts())
    elif style.lower() == 'late-spring':
        # lot-sat. warming
        # mid-autumn
        img = apply_look_up_table(img, get_spring_luts())
        img = apply_look_up_table(img, get_autumn_luts())

    # Experimental filters; non-examinable content
    elif style.lower() == 'nuclear-waste':
        blut = create_look_up_table([(0, 0), (20, 0), (200, 140), (255, 255)])
        glut = create_look_up_table([(0, 100), (80, 180), (255, 255)])
        rlut = create_look_up_table([(0, 30), (200, 180), (255, 255)])
        img = apply_look_up_table(img, (blut, glut, rlut))
    elif style.lower() == 'ocean-waves':
        blut = create_look_up_table([(0, 120), (50, 160), (105, 198), (145, 215), (190, 230), (255, 255)])
        glut = create_look_up_table([(0, 0), (22, 60), (125, 180), (255, 255)])
        rlut = create_look_up_table([(0, 50), (40, 60), (80, 102), (122, 148), (185, 185), (255, 210)])
        img = apply_look_up_table(img, (blut, glut, rlut))
    elif style.lower() == 'neon-blaze':
        blut = create_look_up_table([(0, 80), (40, 90), (80, 102), (122, 148), (185, 185), (255, 210)])
        glut = create_look_up_table([(0, 0), (10, 0), (22, 50), (125, 100), (255, 255)])
        rlut = create_look_up_table([(0, 130), (50, 160), (105, 198), (145, 215), (190, 230), (255, 255)])
        img = apply_look_up_table(img, (blut, glut, rlut))
    else:
        print("Filter unrecognised.")
    img = apply_look_up_table(img, (blut, glut, rlut))
    return img


def display_histogram(img: np.ndarray) -> None:
    blue, green, red = cv2.split(img)
    plt.hist(blue.ravel(), 256, [0, 256], color='blue')
    plt.hist(green.ravel(), 256, [0, 256], color='green')
    plt.hist(red.ravel(), 256, [0, 256], color='red')
    plt.show()


def problem3(image_path=IMG_PATH, enforce_img=False, enforced_img=ig, smoothing_coefficient=args.smoothing_coefficient,
             sub_filter=args.sub_filter, contrast_intensity=args.contrast_intensity,
             histogram=args.histogram) -> Union[np.ndarray, None]:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) if not enforce_img else enforced_img
    if image is None:
        print("Error: Image not found.")
        return
        # sys.exit(1)
    smoothing_coefficient = int(smoothing_coefficient)
    if histogram: display_histogram(image)
    bilateral_img = apply_bilateral_filter(image, smoothing_coefficient, 2.0, 25.0)
    out_img = apply_filter_main(bilateral_img, sub_filter, contrast_intensity)
    if histogram: display_histogram(out_img)
    if not args.multiple_filters:
        cv2.imshow('Input image', image)
        if smoothing_coefficient != 0: cv2.imshow('Bilateral smoothing', bilateral_img)
        if sub_filter.lower() != 'none': cv2.imshow(f'{sub_filter}', out_img)
        key = cv2.waitKey(0)
        if key == ord('x'):
            cv2.destroyAllWindows()
    return out_img


# ---------------------------------------- PART 4: FACE SWIRL -------------------------------------------------------- #

def create_butterworth_low_pass_filter(img: np.ndarray, D_0: int, n: float) -> np.ndarray:
    """
    :param img: Input image
    :param D_0: Distance from origin for the "cut-off frequency"
    :param n: Order of filter
    :return: 2-D np array representing the Butterworth low pass filter
    """
    rows, cols = img.shape[0], img.shape[1]
    D = lambda u, v: np.sqrt((u - mid_u) ** 2 + (v - mid_v) ** 2) if u != mid_u or v != mid_v else 1
    H = lambda u, v: 1 / (1 + (D(u, v) / D_0) ** (2 * n))
    mid_u, mid_v = rows // 2, cols // 2
    blp = np.zeros((rows, cols, 2), np.float64)
    for i in range(rows):
        for j in range(cols):
            blp[i, j] = H(i, j)
    return np.float64(blp)


def get_fourier_transform(img: np.ndarray) -> np.ndarray:
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT | cv2.DFT_SCALE)
    dft = np.fft.fftshift(dft)
    return dft


def apply_butterworth_low_pass_filter(img: np.ndarray, blp: np.ndarray) -> np.ndarray:
    dft = get_fourier_transform(img)
    out = np.abs(cv2.idft(np.fft.ifftshift(dft * blp)))
    out = cv2.magnitude(out[:, :, 0], out[:, :, 1])
    return np.uint8(np.clip(out, 0, 255))


def butterworth_low_pass_filter_main(img: np.ndarray, cut_off_distance: int = 80, order: float = 3.0) -> np.ndarray:
    blp = create_butterworth_low_pass_filter(img, cut_off_distance, order)
    filtered = [apply_butterworth_low_pass_filter(channel, blp) for channel in cv2.split(img)]
    return np.uint8(np.clip(cv2.merge(tuple(filtered)), 0, 255))


# Bilinear interpolation: take weighted average of the surrounding four pixels
def apply_bilinear_interpolation(img: np.ndarray, u: float, v: float) -> np.ndarray:
    fu, cu, fv, cv = floor(u), ceil(u), floor(v), ceil(v)
    try: I1 = ((cv-v)/(cv-fv))*img[fu, fv] + ((v-fv)/(cv-fv))*img[fu, cv]
    except ZeroDivisionError: I1 = img[fu, int(v)]
    try: I2 = ((cv-v)/(cv-fv))*img[cu, fv] + ((v-fv)/(cv-fv))*img[cu, cv]
    except ZeroDivisionError: I2 = img[cu, int(v)]
    try: I3 = ((cu-u)/(cu-fu))*I1 + ((u-fu)/(cu-fu))*I2
    except ZeroDivisionError: I3 = I1
    return I3


def apply_nearest_neighbour_interpolation(img: np.ndarray, u: float, v: float) -> np.ndarray:
    return img[round(u), round(v)]


def apply_swirl(img: np.ndarray, mid: Tuple[int, int], radius: int, strength: float, bilinear: bool) -> np.ndarray:
    rows, cols = img.shape[0], img.shape[1]

    def map_polar_coords(u: int, v: int) -> np.ndarray:
        rho = np.sqrt((u - mid_i) ** 2 + (v - mid_j) ** 2)
        if rho < padded_radius and rho != 0:
            s = ((padded_radius - rho) / delta) * strength if padded_radius - rho <= delta else strength
            theta = s * np.exp(-rho / radius) + np.arctan2(v - mid_j, u - mid_i)
            ni = float(np.clip(mid_i + rho * np.cos(theta), 0, rows - 1))
            nj = float(np.clip(mid_j + rho * np.sin(theta), 0, cols - 1))
            return apply_bilinear_interpolation(img, ni, nj) if bilinear \
                else apply_nearest_neighbour_interpolation(img, ni, nj)
        return img[u, v]

    if radius == 0 or strength == 0: return img
    mid_i, mid_j = mid
    out = np.zeros((rows, cols, img.shape[2]), dtype=np.uint8)
    padded_radius = int(2 * radius)
    delta = padded_radius - radius
    for i in range(rows):
        for j in range(cols):
            out[i, j] = map_polar_coords(i, j)
    return out


def problem4(image_path=IMG_PATH, enforce_img=False, enforced_img=ig, centre='default', radius=args.radius,
             strength=args.strength, bilinear_interpolation=args.bilinear, prefilter=args.butterworth,
             cut_off_freq=args.cut_off_frequency, order=args.order) -> Union[np.ndarray, None]:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) if not enforce_img else enforced_img
    if image is None:
        print("Error: Image not found.")
        return
        # sys.exit(1)
    radius = int(radius // 2)
    cent = (image.shape[0]/2, image.shape[1]/2) if centre == 'default' else centre
    butter = butterworth_low_pass_filter_main(image, cut_off_freq, order) if prefilter else image
    swirl = apply_swirl(butter, cent, radius, strength, bilinear_interpolation)
    if not args.multiple_filters:
        iswirl = apply_swirl(swirl, cent, radius, -strength, bilinear_interpolation)
        difference = image - iswirl
        bdifference = butter - iswirl

        cv2.imshow("Input image", image)
        if prefilter: cv2.imshow("Butterworth Low-pass Pre-filter", butter)
        cv2.imshow("Swirled Image", swirl)
        cv2.imshow("Reversal of the Swirl", iswirl)
        cv2.imshow("Subtraction between original image and reversed swirl", difference)
        if prefilter: cv2.imshow("Subtraction between low-pass filtered image and reversed swirl", bdifference)

        key = cv2.waitKey(0)
        if key == ord('x'):
            cv2.destroyAllWindows()

    return swirl


# ----------------------------------------- PROCESS ARGS ------------------------------------------------------------- #

for filter_name in INPUT_FILTERS:
    if ig is None:
        print("Error: Image not found.")
        sys.exit(1)

    if filter_name == 'problem1' or filter_name == 'light-leak':
        ig = problem1(enforce_img=True, enforced_img=ig)

    elif filter_name == 'problem2' or filter_name == 'pencil':
        ig = problem2(enforce_img=True, enforced_img=ig)

    elif filter_name == 'problem3' or filter_name == 'beautify':
        ig = problem3(enforce_img=True, enforced_img=ig)

    elif filter_name == 'problem4' or filter_name == 'swirl':
        ig = problem4(enforce_img=True, enforced_img=ig)

    else:
        print("error")
        sys.exit(1)
else:
    if args.multiple_filters:
        if args.out_file == 'n/a':
            cv2.imshow("Filtered Image", ig)
            # exit image upon pressing x
            wkey = cv2.waitKey(0)
            if wkey == ord('x'):
                cv2.destroyAllWindows()
        else:
            cv2.imwrite(args.out_file, ig)

