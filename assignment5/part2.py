import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import math

def compute_errors(original, images):
    errors = [np.abs(original - image).mean() for image in images]
    relative_errors = [(error / errors[0]) for error in errors]

    return relative_errors

def gaussian_filter(image, iterations = 32):
    image_to_blur = image.copy()
    standard_deviation = 0.5
    size = 2 * math.ceil(3*standard_deviation) + 1

    images = [image_to_blur]

    for i in range(32):
        image_to_blur = cv2. GaussianBlur(image_to_blur, (size, size), standard_deviation, borderType = cv2.BORDER_REFLECT)
        images.append(image_to_blur.copy())

    return images

def main():

    I_orig_img = Image.open('lotr.jpg')
    I_orig_img = I_orig_img.convert('L')
    I_orig = np.array(I_orig_img)/255.0

    h, w = np.shape(I_orig)
    
    # Add noise
    gauss = np.random.normal(0, 0.22, (h, w))
    gauss = gauss.reshape(h, w)
    I_n = I_orig + gauss

    # Visualize
    I_orig_img.show()
    Image.fromarray((np.clip(I_n*255.0, 0, 255)).astype(np.uint8)).show()

    # Dummy error visualization
    ratios = np.linspace(0, 1, 10)
    errors = compute_errors(I_orig, [(1 - r) * I_n + r * I_orig for r in ratios])
    plt.plot(ratios, errors)
    plt.show()

    # apply gaussian filter
    iterations = 32
    images_filter = gaussian_filter(I_n, iterations)
    Image.fromarray((np.clip(images_filter[8]*255.0, 0, 255)).astype(np.uint8)).show()
    Image.fromarray((np.clip(images_filter[16]*255.0, 0, 255)).astype(np.uint8)).show()
    Image.fromarray((np.clip(images_filter[24]*255.0, 0, 255)).astype(np.uint8)).show()
    Image.fromarray((np.clip(images_filter[32]*255.0, 0, 255)).astype(np.uint8)).show()
    iter_indices = np.linspace(0, 32, 1)
    error_filter = compute_errors(I_orig, images_filter)
    plt.plot(iter_indices, error_filter)
    plt.show()

if __name__ == "__main__":
    main()