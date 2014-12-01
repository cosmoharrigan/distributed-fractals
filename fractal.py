"""
Distributed computation of the Mandelbrot Set using IPython Parallel

Author: Cosmo Harrigan (based on code by Jake Vanderplas)
"""

import numpy as np
import matplotlib.pyplot as plt


def mandel(x, y, max_iter):
    """
    Given z = x + iy and max_iter, determine whether the candidate
    is in the mandelbrot set for the given number of iterations
    """
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iter):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iter


def compute_column(Ny, ymin, rpart, max_iter, dy):
    """
    Compute one column of the Mandelbrot set
    """
    vector = np.zeros(Ny, dtype=float)
    for y in range(Ny):
        ipart = ymin + y * dy
        color = mandel(rpart, ipart, max_iter)
        vector[y] = color
    return vector


def compute_region(chunk, num_chunks, Nx, Ny, xmin, xmax, ymin, ymax, max_iter):
    """
    Compute multiple columns, each of height Ny, of the Mandelbrot set
    The number of columns computed is: (Nx / num_chunks)
    The columns represent the region with the x-offset of (chunk * cols_per_chunk)
    """

    cols_per_chunk = Nx / num_chunks

    dx = (xmax - xmin) * 1. / Nx
    dy = (ymax - ymin) * 1. / Ny

    result = np.zeros((Ny, cols_per_chunk), dtype=float)
    for x in range(chunk * cols_per_chunk, chunk * cols_per_chunk + cols_per_chunk):
        rpart = xmin + x * dx
        local_index = x - chunk * cols_per_chunk
        result[:, local_index] = compute_column(Ny, ymin, rpart, max_iter, dy)
    return result


def create_fractal(Nx, Ny, xmin, xmax, ymin, ymax, max_iter):
    """Create and return a fractal image"""

    # make an np array of dimension Ny by Nx
    image = np.zeros((Ny, Nx), dtype=float)

    # find the coarse graining (granularity) of the step sizes
    #dx = (xmax - xmin) * 1. / Nx
    #dy = (ymax - ymin) * 1. / Ny

    num_chunks = 12

    # Only 'a' will be an iterable, and the other parameters will remain the same, so in order to use map,
    # we need to wrap the compute_region function in a one-parameter lambda function:
    compute_region_lambda = lambda chunk: compute_region(chunk, num_chunks, Nx, Ny, xmin, xmax, ymin, ymax, max_iter)

    regions = map(compute_region_lambda, range(num_chunks))

    combined = np.hstack(regions)
    plt.imshow(combined, cmap=plt.cm.jet)
    plt.show()

    return image

if __name__ == "__main__":
    # Choose the parameters
    Nx = 300
    Ny = 200
    xmin = -2
    xmax = 1
    ymin = -1
    ymax = 1
    max_iter = 20

    image = create_fractal(Nx, Ny, xmin, xmax, ymin, ymax, max_iter)

    print image.shape
    print image.max()
