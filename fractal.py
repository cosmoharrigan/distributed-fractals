"""
Distributed computation of the Mandelbrot Set using IPython Parallel

Author: Cosmo Harrigan, based on Mandelbrot code by Jake Vanderplas
"""

import numpy as np


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
    The columns describe the region with the x-offset of (chunk / num_chunks) * Nx
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
