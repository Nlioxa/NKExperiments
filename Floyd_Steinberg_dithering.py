# libs
from numpy import array
from imageio import imread, imwrite
from matplotlib import pyplot as plt

# read the source image (as a numpy array)
src_img = imread(R"src_img.jpg")

# function to convert a colored image to grayscale one
def gray_scale(image, convert_type):
    grayscale_image = image.copy()
    if convert_type in range(1, 4):
        [width, height, colors] = image.shape
        for i in range(0, width):
            for j in range(0, height):
                pix = image[i, j]
                if convert_type is 1:
                    ## 1. The lightness method averages the most prominent and least prominent colors: 
                    # →(max(R, G, B) + min(R, G, B)) / 2.
                    gray_scale_value = (min(pix) + max(pix)) / 2
                elif convert_type is 2:
                    ## 2. The average method simply averages the values: 
                    # →(R + G + B) / 3.
                    gray_scale_value = pix.sum() / 3
                elif convert_type is 3:
                    ## 3. The luminosity method is a more sophisticated version of the average method.
                    ## We’re more sensitive to green than other colors, so green is weighted most heavily:
                    # → 0.21 R + 0.72 G + 0.07 B.
                    gray_scale_value = 0.21*pix[0] + 0.72*pix[1] + 0.07*pix[2]
                grayscale_image[i, j] = [int(gray_scale_value) for i in range(colors)]
    return grayscale_image

# function to dither an image with Floyd–Steinberg dithering algorithm
def floyd_steinberg_dithering(image, scale):
    # deep copy of an image
    new_img = image.copy()
    # image shape
    [width, height, colors] = new_img.shape
    # max color value
    max = new_img.max()
    for y in range(0, height - 1):
        for x in range(1, width - 1):
            old_pix = new_img[y, x]
            new_pix = array([
                round(color_value * scale / max) * round(max / scale)
                for color_value in old_pix])
            err_pix = old_pix - new_pix
            new_img[y, x] = new_pix
            new_img[y, x + 1] = [
                new_img[y, x + 1][i] + (err_pix[i] * 7/16)
                for i in range(colors)]
            new_img[y, x - 1] = [
                new_img[y, x - 1][i] + (err_pix[i] * 3/16)
                for i in range(colors)]
            new_img[y + 1, x] = [
                new_img[y + 1, x][i] + (err_pix[i] * 5/16)
                for i in range(colors)]
            new_img[y + 1, x + 1] = [
                new_img[y + 1, x + 1][i] + (err_pix[i] * 1/16)
                for i in range(colors)]
    return new_img


# test the implementation
image = gray_scale(src_img, 3)
new_img = floyd_steinberg_dithering(image, 2)

# compare the source image with the dithered image
plt.figure(figsize=(25, 25))
plt.subplot(221)
plt.imshow(image)
plt.title("Source Image")
plt.subplot(222)
plt.imshow(new_img)
plt.title("Dithered Image")

# save the dithered image
imwrite(R"new_img.jpg", new_img)