"""
CS131 - Computer Vision: Foundations and Applications
Assignment 2
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/18/2017
Python Version: 3.5+

Completed by: Jennifer Moore (jlmoore@stanford.edu)
"""

import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    flip_kernel = np.flip(np.flip(kernel, 0), 1)
    for r in range(Hi):
        for c in range(Wi):
            total = np.sum(flip_kernel * padded[r: r+Hk, c: c+Wk]) 
            out[r, c] = total
    

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    k = (size - 1) / 2;
    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1/(2 * np.pi * (sigma **2))) * np.exp(-((i-k)**2 + (j-k)**2)/ float(2 * (sigma**2)))           
           
    ### END YOUR CODE
    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    x_kern = 1/2 *np.array([1, 0, -1]).reshape((1, 3))
    out = conv(img, x_kern)
    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    y_kern = 1/2 * np.array([1, 0, -1]).reshape((3, 1))
    out = conv(img, y_kern)
    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)
    
    ### YOUR CODE HERE
    grad_x = partial_x(img)
    grad_y = partial_y(img)
    
    G = np.sqrt((grad_x**2) + (grad_y**2))
    #arc tan gives radians from -pi to pi, conver 360degrees to 0 degrees bc they are equal
    
    theta = np.rad2deg(np.arctan2(grad_y, grad_x)) % 360
    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))
       

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
 
    ### BEGIN YOUR CODE
    theta = theta % 360
    
    for r in range(H):
        for c in range(W):
            degree = theta[r, c]
        
            if degree == 0 or degree == 180:
                if c == 0:
                    points = [G[r, c+1]]
                elif c == W-1:
                    points = [G[r, c-1]]
                else:
                    points = [G[r, c-1], G[r, c+1]]
            elif degree == 45 or degree == 225:
                if (c == 0 and r == H-1) or (r == 0 and c == W-1):
                    points = [G[r, c]]
                elif r == 0 or c == 0:
                    points = [G[r+1, c+1]]
                elif r == H-1 or c == W-1:
                    points = [G[r-1, c-1]]
                else:
                    points = [G[r-1, c-1], G[r+1, c+1]]
            elif degree == 90 or degree == 270:
                if r == 0:
                    points = [G[r+1, c]]
                elif r == H-1:
                    points = [G[r-1, c]]
                else:
                    points = [G[r-1, c], G[r+1, c]]
            elif degree == 135 or degree == 315:
                if (c == 0 and r == 0) or (c == W-1 and r == H-1):
                    points = [G[r, c]]
                elif r == 0 or c == W-1:
                    points = [G[r+1, c-1]]
                elif c == 0 or r == H-1:
                    points = [G[r-1, c+1]]
                else:
                    points = [G[r-1, c+1], G[r+1, c-1]]
          
            # if its on one of edges, only compare to one point
            if G[r, c] >= np.max(points):
                out[r, c] = G[r, c];
            

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    ### YOUR CODE HERE
    strong_edges = img > high
    weak_edges = (img < high) & (img > low)
    '''
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if img[r, c] > high:
                strong_edges[r,c] = True
            elif img[r, c] > low:
                weak_edges[r, c] = True
'''
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edge = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    toExplore = []
    visited = np.zeros(edges.shape)
    toExplore.append((0,0))
    '''
    while len(toExplore) != 0:
        cur_r, cur_c = toExplore.pop(0)
        #print(cur_r, cur_c)
        if visited[cur_r, cur_c]:
            continue
            
        visited[cur_r, cur_c] = True
        
        possible_links = get_neighbors(cur_r, cur_c, H, W)
  
        for x in possible_links:
            if visited[x] == False:
                toExplore.append(x)
            if (weak_edge[cur_r, cur_c] == True and edges[x] == True):
                edges[cur_r, cur_c] = True
                '''
    for i in indices:
        possible_links = get_neighbors(i[0], i[1], H, W)
        while len(possible_links) != 0:
            cur_r, cur_c = possible_links.pop(0)
            #print(cur_r, cur_c)
            if visited[cur_r, cur_c]:
                continue

            visited[cur_r, cur_c] = True
            connections = get_neighbors(cur_r, cur_c, H, W)
            for x in connections:
                if visited[x] == False and weak_edges[cur_r, cur_c]:
                    possible_links.append(x)
                if (weak_edge[cur_r, cur_c] == True and edges[i[0], i[1]] == True):
                    edges[cur_r, cur_c] = True

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed_img = conv(img, kernel)
    G, theta = gradient(smoothed_img)
    nms = non_maximum_suppression(G, theta)
    
    strong, weak = double_thresholding(nms, high, low)
    edge = link_edges(strong, weak)

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    H, W = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    
    #loop through useful points that are non 0
    for r, c in zip(ys, xs):
        for index in range(num_thetas):
            rho = r * sin_t[index] + c * cos_t[index]
            accumulator[round(rho + diag_len), index] += 1
    
    ### END YOUR CODE

    return accumulator, rhos, thetas
