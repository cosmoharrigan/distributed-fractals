"""
Distributed computation of the Mandelbrot Set using IPython Parallel

Author: Cosmo Harrigan (based on code by Jake Vanderplas)
"""

import numpy as np
import matplotlib.pyplot as plt

Nx = 300
Ny = 200
xmin = -2
xmax = 1
ymin = -1
ymax = 1
max_iter = 20

dx = (xmax - xmin) * 1. / Nx
dy = (ymax - ymin) * 1. / Ny


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
    vector = np.zeros(Ny, dtype=float)
    for y in range(Ny):
        ipart = ymin + y * dy
        color = mandel(rpart, ipart, max_iter)
        vector[y] = color
    return vector


def compute_region(chunk, cols_per_chunk=25):
    """
    Returns an n x m matrix representing m entire columns
    """
    result = np.zeros((Ny, cols_per_chunk), dtype=float)
    #for x in range(cols_per_chunk):
    for x in range(chunk * cols_per_chunk, chunk * cols_per_chunk + cols_per_chunk):
        rpart = xmin + x * dx    # (chunk * cols_per_chunk) * dx  # replaced x with pixels_per_chunk * chunk
        local_index = x - chunk * cols_per_chunk
        result[:, local_index] = compute_column(Ny, ymin, rpart, max_iter, dy)
        # note that, although this is indexed by x, it is actually going to be offset
    return result


def create_fractal(Nx, Ny, xmin, xmax, ymin, ymax, max_iter):
    """Create and return a fractal image"""

    # make an np array of dimension Ny by Nx
    image = np.zeros((Ny, Nx), dtype=float)

    # find the coarse graining (granularity) of the step sizes
    dx = (xmax - xmin) * 1. / Nx
    dy = (ymax - ymin) * 1. / Ny

    # split 300 into 12 25-piece parts
    total_cols = 300
    num_chunks = 12
    cols_per_chunk = total_cols / num_chunks

    # todo: support passing the number of columns per chunk
    regions = map(compute_region, range(num_chunks))

    combined = np.hstack(regions)
    plt.imshow(combined, cmap=plt.cm.jet)
    plt.show()

    return image

if __name__ == "__main__":
    image = create_fractal(300, 200, -2, 1, -1, 1, 20)  # 300, 200, -2, 1, -1, 1, 20

    print image.shape
    print image.max()
