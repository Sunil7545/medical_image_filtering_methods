import matplotlib.pyplot as plt
import numpy as np
import cv2


def log_transformation(input_image):
    # log transformation of the input image
    return np.log1p(input_image).astype(np.float32)


def mean_filtering(input_image, window_size=(13, 13)):
    return cv2.blur(input_image, window_size)


def gaussian_filtering(input_image, window_size=(13, 13), sigma_x=6, sigma_y=6):
    return cv2.GaussianBlur(input_image, window_size, sigma_x, sigma_y)


def median_filtering(input_image, window_size=13):
    return cv2.medianBlur(input_image, window_size)


def bilateral_filtering(input_image, window_size=13, sigma_color=3, sigma_space=5):
    return cv2.bilateralFilter(input_image, window_size, sigma_color, sigma_space)


def exp_transformation(input_image):
    # exponential transformation
    return np.expm1(input_image)


def signal_to_noise(image_input, axis=0, ddof=0):
    a = np.asanyarray(image_input)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.mean(np.where(sd == 0, 0, m/sd))


def image_filtering_anisotropic():
    """
    Anisotropic image filtering methods
    """

    image_retina = plt.imread("retina_input.jpg")
    print("Input Image SNR: " + str(signal_to_noise(image_retina)))

    fig, axs = plt.subplots(3, 1, figsize=(4, 8))
    axs[0].imshow(image_retina, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title("Input")

    log_input = log_transformation(image_retina)
    median_output = exp_transformation(median_filtering(log_input))
    print("Median SNR: " + str(signal_to_noise(median_output)))
    axs[1].imshow(median_output, cmap='gray', vmin=0, vmax=255)
    axs[1].set_title("Median Filtering")

    bil_output = exp_transformation(bilateral_filtering(log_input))
    print("Bilateral SNR: " + str(signal_to_noise(bil_output)))
    axs[2].imshow(bil_output, cmap='gray', vmin=0, vmax=255)
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

    image_retina = plt.imread("retina_input.jpg")

    fig, axs = plt.subplots(3, 1, figsize=(4, 8))
    axs[0].imshow(image_retina, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title("Input")

    log_input = log_transformation(image_retina)
    mean_output = exp_transformation(median_filtering(log_input))
    axs[1].imshow(mean_output, cmap='gray', vmin=0, vmax=255)
    axs[1].set_title("Mean Filtering")

    gauss_mean = exp_transformation(bilateral_filtering(log_input))
    axs[2].imshow(gauss_mean, cmap='gray', vmin=0, vmax=255)
    axs[2].set_title("Gaussian Filtering")

    for j in range(3):
        axs[j].set_xticks([])
        axs[j].set_yticks([])
    plt.tight_layout()
    plt.show()


image_filtering_anisotropic()
