import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    pass
    ### END YOUR CODE
    out = io.imread(image_path)
    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = image[start_row:start_row + num_rows, start_col:start_col + num_cols, :]
    pass
    ### END YOUR CODE

    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = np.copy(image)

    ### YOUR CODE HERE
    h, w, d = image.shape
    for r in range(h):
        for c in range(w):
                out[r, c, :] = .5 * (image[r, c, :] ** 2)
    pass
    ### END YOUR CODE

    return out


def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!

    ### YOUR CODE HERE
    for r in range(output_rows):
        for c in range(output_cols):
            input_r = int(r * (input_rows/output_rows))
            input_c = int(c * (input_cols/output_cols))
            output_image[r, c, :] = input_image[input_r, input_c, :]
    pass
    ### END YOUR CODE

    # 3. Return the output image
    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    # Reminder: np.cos() and np.sin() will be useful here!

    ## YOUR CODE HERE
    # rotation matrix to go counterclockwise
    rotmat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    newpoint = np.dot(rotmat, point)
    
    pass
    ### END YOUR CODE
    return newpoint


def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)

    ## YOUR CODE HERE
    for r in range(input_rows):
        for c in range(input_cols):
            #Multiplication Way
            #make the point homogeneous
            point = np.array([r, c, 1])
            transM = np.array([[1, 0, -input_rows/2], [0, 1, -input_rows/2], [0, 0, 1]])
            transP = np.dot(transM, point)
        
            # returns 1d vector, we need homogenous 3d
            rotP = rotate2d(transP[:2].copy(), theta)
            rothomo = np.array([rotP[0], rotP[1], 1])
            
            transBack = np.array([[1, 0, input_rows/2], [0, 1, input_rows/2], [0, 0, 1]])
            
            correctP = np.dot(transBack, rothomo)
            
            
            new_r = int(correctP[0])
            new_c = int(correctP[1])
            
            # Non multiplication way
            #point = np.array([r - input_rows/2, c - input_rows/2])
            #newpoint = rotate2d(point, theta)
            #new_r = int(newpoint[0] + input_rows/2)
            #new_c = int(newpoint[1] + input_cols/2)
            
            if (new_r < 0 or new_r >= input_rows or new_c < 0 or new_c >= input_cols):
                continue
            else:
                output_image[r, c, :] = input_image[new_r, new_c, :]
    pass
    ### END YOUR CODE

    # 3. Return the output image
    return output_image
