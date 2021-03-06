import sys
import cv2
import math
import numpy as np
from src import filter_util
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')


def cross_correlation_2d(img, kernel):
    """
    Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    """
    cc_canvas = filter_util.canvas(img.shape, kernel.shape)
    kernel_size = kernel.shape
    for target_column_idx in range(cc_canvas.shape[1]):
        for target_row_idx in range(cc_canvas.shape[0]):
            source_sample = img[
                target_row_idx:target_row_idx + kernel_size[0],
                target_column_idx:target_column_idx + kernel_size[1]]
            cc_canvas[target_row_idx, target_column_idx] = np.sum(source_sample * kernel)
    img_out = filter_util.handle_edge(cc_canvas, img.shape)
    return img_out


def convolve_2d(img, kernel):
    """
    Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    """
    convolution_kernel = kernel[::-1, ::-1]
    return cross_correlation_2d(img, convolution_kernel)


def gaussian_blur_kernel_2d(sigma, width, height):
    """
    Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions width x height such that convolving it
        with an image results in a Gaussian-blurred image.
    """
    kernel = np.zeros((height, width))
    for r in range(height):
        for c in range(width):
            x = r - (width - 1) / 2
            y = c - (height - 1) / 2
            co = 1 / (2 * math.pi * (math.pow(sigma, 2)))
            ex = math.exp(-1 * (math.pow(x, 2) + math.pow(y, 2)) / (2 * math.pow(sigma, 2)))
            g = co * ex
            kernel[r, c] = g
    return kernel


def low_pass(img, sigma, size):
    """
    Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    """
    out_img = np.zeros(img.shape)
    for layer_idx in xrange(img.shape[2]):
        img_layer = img[:, :, layer_idx]
        out_img[:, :, layer_idx] = convolve_2d(img_layer, gaussian_blur_kernel_2d(sigma, size, size))
    return out_img


def high_pass(img, sigma, size):
    """
    Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    """
    return img - low_pass(img, sigma, size)


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,high_low2, mixin_ratio):
    """
    This function adds two images to create a hybrid image, based on
    parameters specified by the user.
    """
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


