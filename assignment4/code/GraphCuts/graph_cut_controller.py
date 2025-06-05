import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from tkinter import *
from PIL import Image

from graph_cut import GraphCut
from graph_cut_gui import GraphCutGui


class GraphCutController:

    def __init__(self, args):
        self.__init_view(args.autoload)

    def __init_view(self, autoload=None):
        root = Tk()
        root.geometry("700x500")
        self._view = GraphCutGui(self, root)

        if autoload is not None:
            self._view.autoload(autoload)

        root.mainloop()

    # TODO: TASK 2.1
    def __get_color_histogram(self, image, seed, hist_res):
        """
	Compute a color histograms based on selected points from an image
	
	:param image: color image
	:param seed: Nx2 matrix containing the the position of pixels which will be
	            used to compute the color histogram
	:param histRes: resolution of the histogram
	:return hist: color histogram
	"""
        # first extract the seed colors from the image
        seed_colors = image[seed[:, 0], seed[:, 1]]  # (N,3) where each entry is the color of the seed

        # create color histogram with resolution "hist_res", bound the values of each channel within [0,255]
        histogram, bins = np.histogramdd(seed_colors, bins=(hist_res, hist_res, hist_res), range=((0, 255), (0, 255), (0, 255)))

        # smooth the histogram to make it more general
        sigma = 0.1
        smoothed_histogram = ndimage.gaussian_filter(histogram, sigma=sigma)

        # normalize the histogram such that it sums up to 1
        normalized_smoothed_histogram = smoothed_histogram / np.sum(smoothed_histogram)

        return normalized_smoothed_histogram

    # TODO: TASK 2.2
    # Hint: Set K very high using numpy's inf parameter
    def __get_unaries(self, image, lambda_param, hist_fg, hist_bg, seed_fg, seed_bg):
        """

        :param image: color image as a numpy array
        :param lambda_param: lamdba as set by the user
        :param hist_fg: foreground color histogram
        :param hist_bg: background color histogram
        :param seed_fg: pixels marked as foreground by the user
        :param seed_bg: pixels marked as background by the user
        :return: unaries : Nx2 numpy array containing the unary cost for every pixels in I (N = number of pixels in I)
        """
        height = image.shape[0]
        width = image.shape[1]
  
        # plus the offset to avoid empty entries
        offset = 10e-10
        hist_fg += offset
        hist_bg += offset

        # reshape the image into (height*width, 3) array
        hist_res = hist_fg.shape[0]
        reshaped_image = image.reshape(-1, 3)
        r_values = reshaped_image[:, 0]  # (height*width,)
        g_values = reshaped_image[:, 1]  # (height*width,)
        b_values = reshaped_image[:, 2]  # (height*width,)

        # get the r, g, b indices of the histogram bins
        r_indices = np.clip(np.floor(r_values/255 * hist_res), 0, hist_res - 1).astype(int)
        g_indices = np.clip(np.floor(g_values/255 * hist_res), 0, hist_res - 1).astype(int)
        b_indices = np.clip(np.floor(b_values/255 * hist_res), 0, hist_res - 1).astype(int)

        # get Pr(I_p|O)
        probability_fg = hist_fg[r_indices, g_indices, b_indices]
        # get R_p_obj = -lnPr(I_p|O)
        R_p_obj = -np.log(probability_fg)

        # get Pr(I_p|B)
        probability_bg = hist_bg[r_indices, g_indices, b_indices] 
        # get R_p_bkg = -lnPr(I_p|B)
        R_p_bkg = -np.log(probability_bg)

        # the edge weights with the source
        # for p not in O neither B
        R_p_bkg *= lambda_param
        R_p_bkg = R_p_bkg.reshape(height, width)
        # set weights of p in O to inf
        R_p_bkg[seed_fg[:,0], seed_fg[:,1]] = np.inf
        # set weights of p in B to 0
        R_p_bkg[seed_bg[:,0], seed_bg[:,1]] = 0
        R_p_bkg = R_p_bkg.reshape(-1, 1)

        # the edge weights with the sink
        # for p not in O neither B
        R_p_obj *= lambda_param
        R_p_obj = R_p_obj.reshape(height, width)
        # set weights of p in O to 0
        R_p_obj[seed_fg[:,0], seed_fg[:,1]] = 0
        # set weights of p in B to 0
        R_p_obj[seed_bg[:,0], seed_bg[:,1]] = np.inf
        R_p_obj = R_p_obj.reshape(-1, 1)

        unaries = np.column_stack((R_p_bkg, R_p_obj))
        return unaries




    # TODO: TASK 2.3
    # Hint: Use coo_matrix from the scipy.sparse library to initialize large matrices
    # The coo_matrix has the following syntax for initialization: coo_matrix((data, (row, col)), shape=(width, height))
    def __get_pairwise(self, image):
        """
        Get pairwise terms for each pairs of pixels on image
        :param image: color image as a numpy array
        :return: pairwise : sparse square matrix containing the pairwise costs for image
        """
        sigma = 5

        height = image.shape[0]
        width = image.shape[1]

        data = []
        row = []
        column = []

        # use the eight neighbor model for each pixel.
        # to avoid recomputation, for each pixel we add the edge with its right 
        # and below neighbor(except for pixels on the right and bottom boundary)
        # indices calculated rowise as default of reshape method to keep consitency with the _get_unaries()
        for i in range(height):
            for j in range(width):
                index = i*width + j

                # add edge with right neighbor
                if j != (width-1):
                    index_right = i*width + j + 1

                    value_i_j = image[i,j]
                    value_i_j_right_neighbor = image[i, j+1]
                    diff_value = value_i_j - value_i_j_right_neighbor
                    # dist(p,q) in the formula of the paper is ignored, since all distances are 1
                    penalty = np.exp(np.clip(-np.sum(diff_value**2)/(2*sigma**2), -np.inf, 0))

                    data.append(penalty)
                    row.append(index)
                    column.append(index_right)

                    data.append(penalty)
                    row.append(index_right)
                    column.append(index)

                # add edge with below neighbor    
                if i != (height-1):    
                    index_below = (i+1)*width + j

                    value_i_j = image[i,j]
                    value_i_j_below_neighbor = image[i+1, j]
                    diff_value = value_i_j - value_i_j_below_neighbor
                    # dist(p,q) in the formula of the paper is ignored, since all distances are 1
                    penalty = np.exp(np.clip(-np.sum(diff_value**2)/(2*sigma**2), -np.inf, 0))
                    
                    data.append(penalty)
                    row.append(index)
                    column.append(index_below)

                    data.append(penalty)
                    row.append(index_below)
                    column.append(index)

        data = np.array(data)
        row = np.array(row)
        column = np.array(column)

        pairwise  = coo_matrix((data, (row, column)), shape=(height*width, height*width))
        return pairwise

        
    # TODO TASK 2.4 get segmented image to the view
    def __get_segmented_image(self, image, labels, background=None):
        """
        Return a segmented image, as well as an image with new background 
        :param image: color image as a numpy array
        :param label: labels a numpy array
        :param background: color image as a numpy array
        :return image_segmented: image as a numpy array with red foreground, blue background
        :return image_with_background: image as a numpy array with changed background if any (None if not)
        """
        image_segmented = image.copy()
        fg_pixel_x, fg_pixel_y = np.where(labels == False)
        image_segmented[fg_pixel_x, fg_pixel_y] = np.array([255,0,0])
        bg_pixel_x, bg_pixel_y = np.where(labels == True)
        image_segmented[bg_pixel_x, bg_pixel_y] = np.array([0,0,255])

        image_with_background = None
        if background is not None:
            image_with_background = image.copy()
            image_with_background[bg_pixel_x, bg_pixel_y] = background[bg_pixel_x, bg_pixel_y]

        return image_segmented, image_with_background


    def segment_image(self, image, seed_fg, seed_bg, lambda_value, background=None):
        image_array = np.asarray(image)
        background_array = None
        if background:
            background_array = np.asarray(background)

        #print(image_array.shape[0], image_array.shape[1])    
        seed_fg = np.array(seed_fg)
        seed_fg = seed_fg[:, ::-1]
        #print(seed_fg)
        seed_bg = np.array(seed_bg)
        seed_bg = seed_bg[:, ::-1]
        height, width = np.shape(image_array)[0:2]
        num_pixels = height * width

        # TODO: TASK 2.1 - get the color histogram for the unaries
        hist_res = 32
        cost_fg = self.__get_color_histogram(image_array, seed_fg, hist_res)
        cost_bg = self.__get_color_histogram(image_array, seed_bg, hist_res)

        # TODO: TASK 2.2-2.3 - set the unaries and the pairwise terms
        unaries = self.__get_unaries(image_array, lambda_value, cost_fg, cost_bg, seed_fg, seed_bg)
        pairwise = self.__get_pairwise(image_array)

        # TODO: TASK 2.4 - perform graph cut
        # Your code here\
        
        # use 6*num_pixels to bound the number of edges
        # 4*num_pixels for connections between pixels, since each pixel has at most 4 edges
        # respectively num_pixels edges to connect each terminal
        graph = GraphCut(num_pixels, 6*num_pixels) 
        graph.set_unary(unaries)
        graph.set_neighbors(pairwise)

        minCut = graph.minimize()
        labels = graph.get_labeling()
        labels = labels.reshape(height, width)

        # TODO TASK 2.4 get segmented image to the view
        segmented_image, segmented_image_with_background = self.__get_segmented_image(image_array, labels,
                                                                                      background_array)
        # transform image array to an rgb image
        segmented_image = Image.fromarray(segmented_image, 'RGB')
        self._view.set_canvas_image(segmented_image)
        if segmented_image_with_background is not None:
            segmented_image_with_background = Image.fromarray(segmented_image_with_background, 'RGB')
            plt.imshow(segmented_image_with_background)
            plt.show()
