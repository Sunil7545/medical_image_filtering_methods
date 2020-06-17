import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import io
from skimage.util import compare_images


def log_transformation(input_image):
    # log transformation of the input image
    return np.log1p(input_image).astype(np.float32)


def mean_filtering(input_image, window_size=(15, 15)):
    filtered_image = cv2.blur(input_image, window_size)
    filtered_image = np.nan_to_num(filtered_image)
    return filtered_image


def gaussian_filtering(input_image, window_size=(15, 15), sigma_x=6, sigma_y=8):
    filtered_image = cv2.GaussianBlur(input_image, window_size, sigma_x, sigma_y)
    filtered_image = np.nan_to_num(filtered_image)
    return filtered_image


def median_filtering(input_image, window_size=15):
    filtered_image = np.array(cv2.medianBlur(input_image, window_size))
    filtered_image = np.nan_to_num(filtered_image)
    return filtered_image


def bilateral_filtering(input_image, window_size=13, sigma_color=3, sigma_space=5):
    filtered_image = cv2.bilateralFilter(input_image, window_size, sigma_color, sigma_space)
    filtered_image = np.nan_to_num(filtered_image)
    return filtered_image


def exp_transformation(input_image):
    # exponential transformation
    filtered_image = np.expm1(input_image)
    return filtered_image*(255/np.max(filtered_image))


def signal_to_noise(image_input, axis=0, ddof=0):
    a = np.asanyarray(image_input)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.mean(np.where(sd == 0, 0, m/sd))


def speckle_generator(image):
    row, col = image.shape
    gauss = np.random.randn(row, col)
    gauss = gauss.reshape(row, col)
    noisy = image + 0.4*image * gauss
    return noisy*(255/np.max(image))


def image_filtering_anisotropic():
    """
    Anisotropic image filtering methods
    """

    # reading, corrupting image with noise, and removing nan, negative values.
    image_ref = io.imread('input.jpg', as_gray=True)
    image_retina = speckle_generator(image_ref)
    image_retina = np.nan_to_num(image_retina)
    image_retina[image_retina < 0] = 1
    print("Input Image SNR: " + str(signal_to_noise(image_retina)))

    fig, axs = plt.subplots(3, 1, figsize=(4, 9.5))
    axs[0].imshow(image_retina, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title("Input")

    log_input = log_transformation(image_retina)
    median_output = exp_transformation(median_filtering(log_input))
    compute_diff = compare_images(image_retina, median_output, method='checkerboard', n_tiles=(4, 4))
    io.imsave("diff_input_output.jpg", compute_diff)
    print("Median SNR: " + str(signal_to_noise(median_output)))
    axs[1].imshow(median_output, cmap='gray')
    axs[1].set_title("Median Filtering")

    bil_output = exp_transformation(bilateral_filtering(log_input))
    print("Bilateral SNR: " + str(signal_to_noise(bil_output)))
    axs[2].imshow(bil_output, cmap='gray')
    axs[2].set_title("Bilateral Filtering")

    for j in range(3):
        axs[j].set_xticks([])
        axs[j].set_yticks([])
    plt.tight_layout()
    plt.show()


def image_filtering_isotropic():
    """
    Isotropic methods for image filtering
    """

    # image_retina = plt.imread("retina_input.jpg")
    image_ref = io.imread('input.jpg', as_gray=True)
    image_retina = speckle_generator(image_ref)
    image_retina = np.nan_to_num(image_retina)
    image_retina[image_retina < 0] = 1

    fig, axs = plt.subplots(3, 1, figsize=(4, 9.5))
    axs[0].imshow(image_retina, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title("Input")

    log_input = log_transformation(image_retina)
    mean_output = exp_transformation(mean_filtering(log_input))
    axs[1].imshow(mean_output, cmap='gray')
    axs[1].set_title("Mean Filtering")

    gauss_mean = exp_transformation(bilateral_filtering(log_input))
    axs[2].imshow(gauss_mean, cmap='gray')
    axs[2].set_title("Gaussian Filtering")

    for j in range(3):
        axs[j].set_xticks([])
        axs[j].set_yticks([])
    plt.tight_layout()
    plt.show()


image_filtering_anisotropic()
