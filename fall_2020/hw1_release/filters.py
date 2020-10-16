"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+

Completed by: Jennifer Moore (jlmoore@stanford.edu)
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    dif_h = int(Hk/2)
    dif_w = int(Hk/2)
    
    ### YOUR CODE HERE
    for r in range(Hi):
        for c in range(Wi):
            total = 0
            for kr in range(Hk):
                for kc in range(Wk):
                    if r+dif_h-kr < 0 or c+dif_w-kc < 0 or r+dif_h-kr >= Hi or c+dif_w-kc >= Wi:
                        total += 0
                    else:
                        total += kernel[kr, kc] * image[r+dif_h-kr, c+dif_w-kc]
            out[r, c] = total
               
    pass
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """
    
    H, W = image.shape
    out = None
  
    ### YOUR CODE HERE
    out = np.zeros((H+2*pad_height, W+2*pad_width))
    out[pad_height: H+pad_height, pad_width: W+pad_width] = image
    pass
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    # divide by two instead of -1 because need the first r,c in the image to start in MIDDLE of kernel to keep image same size
    padded = zero_pad(image, int(Hk/2), int(Wk/2))
    flip_kernel = np.flip(np.flip(kernel, 0), 1)
    for r in range(Hi):
        for c in range(Wi):
            total = np.sum(flip_kernel * padded[r: r+Hk, c: c+Wk]) 
            out[r, c] = total
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    ##So we can use conv_fast function which flips it back to normal
    flip_temp = np.flip(np.flip(g, 0), 1)
    out = conv_fast(f, flip_temp)
    
    pass
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    no_mean = g - np.mean(g)
    out = cross_correlation(f, no_mean)
    pass
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    Hi, Wi = f.shape
    Ht, Wt = g.shape
    out = np.zeros((Hi, Wi))
    f = zero_pad(f, int(Ht/2), int(Wt/2))
    
    for r in range(Hi):
        for c in range(Wi):
            f2 = (f[r:r+Ht, c:c+Wt] - np.mean(f[r: r+Ht, c: c+Wt])) / np.std(f[r: r+Ht, c: c+Wt])
            g2 = (g - np.mean(g))/ np.std(g)
            
            out[r, c] = np.sum(f2 * g2) 
    pass
    ### END YOUR CODE

    return out
