# Written by Avinoam Nukrai
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import convolve
from skimage.color import rgb2gray
import imageio as iio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio.v2 as imageio


# Constants
EXPAND_FACTOR = 2
GRAYSCALE = 1
RGB = 2
RGB_DIM = 3
MAX_GS = 255


def get_blur_image(im, blur_filter):
    """
    This function blurring a given image by calculating of convolution with
    the given blur_filter.
    :param im: the image to blur
    :param blur_filter:  the filter to apply on the image (a vector)
    :return: a blur image
    """
    blured_image_x_axis = convolve(im, blur_filter)    # blur rows (X-axis)
    final_blured_image = convolve(blured_image_x_axis, blur_filter.transpose())  # blur the columns (Y-axis)
    return final_blured_image


def pad_image_with_zeros(im):
    """
    This function taking an image and padding it with zeros - add zero between
    each 2 pixels in the given image.
    :param im: a given image to pad
    :return: new image, in size = 2 * size(given image) with zero in each 2nd pixel
    """
    pad_image = np.zeros((EXPAND_FACTOR * im.shape[0], EXPAND_FACTOR * im.shape[1]), dtype=im.dtype)    # creating a 2d zero matrix, 2 times size of the original image size
    pad_image[::2, ::2] = im    # replace every 2nd zero pixel with pixel from original image
    return pad_image


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    # first step - to blur the image (do convolution with the filter_vec)
    blured_image = get_blur_image(im, blur_filter)
    # second step - drop every 2nd pixel (reduce the image size in half)
    reduced_image = blured_image[::2, ::2]
    return reduced_image


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    # first step - to pad the given image with zeros
    padded_image = pad_image_with_zeros(im)
    # second step - blur the new image (for normalize the zero's pixels effect)
    expand_image = get_blur_image(padded_image, blur_filter)
    return expand_image


def create_filter(filter_size):
    """
    This function creates a gaussian filter, given its size
    :param filter_size: the wanted size of the gaussian filter
    :return: gaussian filter
    """
    gaussian_filter = conv = np.ones(2)
    for i in range(filter_size - 2):
        gaussian_filter = np.convolve(gaussian_filter, conv)
    return (1 / (2 ** (filter_size - 1))) * gaussian_filter.reshape(1, filter_size)     # returning the filter vector after normalize it and reshape to row vector


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in cons  tructing the pyramid filter
    :return: pyr, gaussian_filter. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and gaussian_filter is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    pyr = [im]
    gaussian_filter = create_filter(filter_size)
    prev_gaussian = im
    for i in range(1, max_levels):
        if prev_gaussian.shape == (16, 16):  # validate the min size of gaussian
            break
        gaussian_im = reduce(prev_gaussian, gaussian_filter)
        pyr.append(gaussian_im)
        prev_gaussian = gaussian_im
    # display_pyramid(pyr, 10, False)
    return pyr, gaussian_filter


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, gaussian_filter. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and gaussian_filter is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    gaussian_pyramid, gaussian_filter = build_gaussian_pyramid(im,  max_levels, filter_size)
    # Ln = Gn - Expand{Gn+1}
    pyr = [np.subtract(gaussian_pyramid[i], expand(gaussian_pyramid[i + 1], 2 * gaussian_filter)) for i in range(len(gaussian_pyramid) - 1)]
    pyr.append(gaussian_pyramid[-1])
    # display_pyramid(pyr, 10, True)
    return pyr, gaussian_filter


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """
    # first step - to expand all the laplacian images in the pyramid
    expand_filter_vec = 2 * filter_vec
    expanded_lap_pyr = []
    mult_lpyr = []
    for lap_pyr in lpyr:
        while lap_pyr.shape != lpyr[0].shape:  # means that there is more to expand!
            lap_pyr = expand(lap_pyr, expand_filter_vec)
        expanded_lap_pyr.append(lap_pyr)
    # second step - to mult the expand lap images in the right coeff
    for (idx, expand_lpyr) in enumerate(expanded_lap_pyr):
        mult_lpyr.append(expand_lpyr * coeff[idx])
    # return the sum of all the final images (this is the final original image)
    return sum(mult_lpyr)


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
    from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    max_pyr_height = pyr[0].shape[0]
    pyr = pyr[:levels]
    for i in range(len(pyr)):   # padding with zeros all the "free" pixels
        curr_pyr_height = pyr[i].shape[0]
        pyr[i] = pyr[i].clip(0, 1)  # stretch the values into the interval (0, 1)
        pyr[i] = pyr[i] - np.min(pyr[i]) / (np.max(pyr[i] - np.min(pyr[i])))  # Linear stretch of each pixel in each pyramid in pyr, we want that the maximal value will be closest to 1 when the minimum value will be closest to 0
        pyr[i] = np.pad(pyr[i], [(0, max_pyr_height - curr_pyr_height), (0, 0)], 'constant', constant_values=0)
    return np.hstack(pyr)   # return all the pyramids in a stack representation


def display_pyramid(pyr, levels, lap):
    """
    display the rendered pyramid
    :param pyr: the pyramid we want to display
    :param levels: the level of which we want to present fron the given pyramid
    :return: none, just show the pyramid
    """
    rendered_pyramid = render_pyramid(pyr, levels)
    plt.imshow(rendered_pyramid)
    if lap:
        plt.title('Laplacian Pyramid')
    else:
        plt.title('Gaussian Pyramid')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    # 1. Construct Laplacian pyramids L1 and L2 for the input images im1 and im2, respectively.
    im1_laplacian_pyr, filter_vec1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    im2_laplacian_pyr, filter_vec2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    # 2. Construct a Gaussian pyramid Gm for the provided mask (convert it first to np.float64).
    mask = np.float64(mask)
    mask_gaussian = build_gaussian_pyramid(mask, max_levels, filter_size_mask)[0]
    # 3. Construct the Laplacian pyramid Lout: Lout[k] = Gm[k] · L1[k] + (1 − Gm[k]) · L2[k]
    laplacian_out = []
    for k in range(len(im1_laplacian_pyr)):
        laplacian_out.append(mask_gaussian[k] * im1_laplacian_pyr[k] +
                             (1 - mask_gaussian[k]) * im2_laplacian_pyr[k])
    # 4. Reconstruct the resulting blended image from the Laplacian pyramid Lout (using ones for coefficients).
    # coeff = [1] * im2_laplacian_pyr[0].shape[0]
    coeff = [1,1,1, 1, 0]
    final_blended_image = laplacian_to_image(laplacian_out, filter_vec1, coeff)
    return final_blended_image


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    # We have 3 options only of converting: rgb -> rgb, gs -> gs, rgb -> gs
    image = iio.v3.imread(filename)
    if (len(image.shape) == RGB_DIM and representation == RGB) or \
            (len(image.shape) != RGB_DIM and representation == GRAYSCALE):
        # rgb -> rgb or gs -> gs
        return np.float64(image / MAX_GS)
    elif len(image.shape) == RGB_DIM and representation == GRAYSCALE:
        # rgb -> gs
        return rgb2gray(image)


def relpath(filename):
    """
    This function opens file in external folder
    :param filename: the file we want to open
    :return: the translated path
    """
    return os.path.join(os.path.dirname(__file__), filename)


def rgb_images_blending(image1, image2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    This function responsible for blending 2 RGB images, doing blending
    separately for each color (red, green and blue) and then stack all together
    :param image1: first image to blend
    :param image2: second image to blend
    :param mask: the mask to blend with
    :param max_levels:max levels of blending the images
    :param filter_size_im: images filter size
    :param filter_size_mask: max filter size
    :return:
    """
    r1, g1, b1 = image1[:, :, 0], image1[:, :, 1], image1[:, :, 2]
    r2, g2, b2 = image2[:, :, 0], image2[:, :, 1], image2[:, :, 2]
    r_blend = pyramid_blending(r1, r2, mask, max_levels, filter_size_im, filter_size_mask)
    g_blend = pyramid_blending(g1, g2, mask, max_levels, filter_size_im, filter_size_mask)
    b_blend = pyramid_blending(b1, b2, mask, max_levels, filter_size_im, filter_size_mask)
    final_blended_rgb = np.dstack((r_blend, g_blend, b_blend))
    final_blended_rgb[final_blended_rgb > 1] = 1
    return final_blended_rgb


def plot_with_figure(figure, image, image_details):
    """
    This function plots a given image to a given figure
    :param figure: figure to plot into
    :param image: image to plot
    :param image_details: details of the image - name, and number
    :return: none
    """
    figure.add_subplot(2, 2, image_details[1])
    plt.title(image_details[0], loc='center')
    plt.imshow(image, cmap='gray')


def plot_results1(image1, image2, mask, blend_image):
    """
    This function plots the results of the RGB blending I've implemented in
    the examples functions.
    :param image1: first image to plot
    :param image2: second image to plot
    :param mask: mask to blend with
    :param blend_image: the final blend image
    :return: none
    """
    f = plt.figure()
    plot_with_figure(f, image1, ["Image 1", 1])
    plot_with_figure(f, image2, ["Image 2", 2])
    plot_with_figure(f, mask, ["", 3])
    plot_with_figure(f, blend_image, ["", 4])
    plt.show()


def blending_example1():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
    and out the blended image
    """
    image1 = read_image(relpath("externals/apple.jpg"), RGB)
    image2 = read_image(relpath("externals/orange.jpg"), RGB)
    mask = (1 - read_image(relpath("externals/mask5.jpg"), GRAYSCALE)).astype(np.bool_)
    blend_image = rgb_images_blending(image1, image2, mask, 5, 10, 50)
    plot_results(image1, image2, mask, (blend_image * MAX_GS).astype(np.uint8))
    return image1, image2, mask, blend_image


def blending_example2():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
    and out the blended image
    """
    image1 = read_image(relpath("externals/avi4.jpg"), RGB)
    image2 = read_image(relpath("externals/dave.jpg"), RGB)
    mask = (1 - read_image(relpath("externals/mask3.jpg"), GRAYSCALE)).astype(np.bool_)
    blend_image = rgb_images_blending(image1, image2, mask, 1000, 10, 2)
    plot_results(image1, image2, mask, (blend_image * MAX_GS).astype(np.uint8))
    return image1, image2, mask, blend_image


def create_hybrid_image(img1_path, img2_path, max_levels, filter_size_im):
    """
    Create a hybrid image by combining Laplacian pyramids of two input images.
    :param image1_path: File path of the first input image
    :param image2_path: File path of the second input image
    :param max_levels: Max levels for the pyramids
    :param filter_size_im: Size of the Gaussian filter for images
    :return: The blended hybrid image
    """
    img1 = read_image(img1_path, GRAYSCALE)
    img2_path = read_image(img2_path, GRAYSCALE)
    # Build Laplacian pyramids for the two images
    pyramid1, _ = build_laplacian_pyramid(img1, max_levels, filter_size_im)
    pyramid2, _ = build_laplacian_pyramid(img2_path, max_levels, filter_size_im)
    # Combine Laplacian pyramids to create the hybrid image
    hybrid_pyramid = pyramid1[:2] + pyramid2[2:]
    # Reconstruct the hybrid image from the combined pyramid
    hybrid_img = laplacian_to_image(hybrid_pyramid, create_filter(filter_size_im), [1] * len(hybrid_pyramid))
    # Normalize the hybrid image
    hybrid_img = (hybrid_img - np.min(hybrid_img)) / (np.max(hybrid_img) - np.min(hybrid_img))
    # Create the actual hybrid image
    hybrid_img = (hybrid_img * MAX_GS).astype(np.uint8)
    return hybrid_img


def hybrid_example(img1_path, img2_path, max_levels, filter_size_im):
    """
    Create a hybrid image by combining Laplacian pyramids of two input images.
    :param image1_path: File path of the first input image
    :param image2_path: File path of the second input image
    :param max_levels: Max levels for the pyramids
    :param filter_size_im: Size of the Gaussian filter for images
    :return: None, just plot the hybrid image
    """
    hybrid_img = create_hybrid_image(img1_path, img2_path, max_levels, filter_size_im)
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    # Load images
    img1 = rgb2gray(mpimg.imread(img1_path))
    img2 = rgb2gray(mpimg.imread(img2_path))
    # Plot each image
    axs[0].imshow(img1, cmap='gray')
    axs[0].set_title('Avinoam')
    axs[1].imshow(img2, cmap='gray')
    axs[1].set_title('Avinoam Brother')
    axs[2].imshow(hybrid_img, cmap='gray')
    axs[2].set_title('Hybrid Image')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()



def plot_results(img1, img2, img3, blended_result):
    # Create a figure and axis
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    # Plot each image
    axs[0].imshow(img1)
    axs[0].set_title('Image 1')
    axs[1].imshow(img2)
    axs[1].set_title('Image 2')
    axs[2].imshow(img3, cmap='gray')
    axs[2].set_title('Mask')
    axs[3].imshow(blended_result)
    axs[3].set_title('Blended Result')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Paths to the input images
    image1_path = 'avi4.jpg'
    image2_path = 'david4.jpg'
    # # blending
    # gaussian_filter = conv = np.ones(2)
    # for i in range(5):
    #     gaussian_filter = np.convolve(gaussian_filter, conv)
    #     print(gaussian_filter)
    # arr = np.asarray([[1,2,3,4], [5,6,7,8]])
    # np.apply_along_axis()
    # mat = np.asarray([[1,2,3], [0, -1, 1]])
    # print(np.min(mat))
    blending_example1()
    # blending_example2()
    # # hybrid image
    # hybrid_example(image1_path, image2_path, 200, 30)
    pass
