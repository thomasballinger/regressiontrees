"""Really easy synthetic data

Generate "eyes," 20x20 matrices of grayscale values
Figure out where the center is in them
"""



import random
import numpy
from pylab import imshow
import matplotlib.pyplot

def synthetic(x, y, width=100, height=100, radius=20):
    in_circle = lambda xi, yi: ((x-xi)**2 + (y-yi)**2) < radius**2
    return numpy.random.rand(height, width) + numpy.array([[1 if in_circle(i, j) else 0
                                                            for i in range(width)]
                                                           for j in range(height)])

training = [(synthetic(x, y), (x, y)) for x, y in [(random.random(), random.random()) for _ in range(100)]]

def show(a2d):
    imshow(a2d)
    matplotlib.pyplot.show()

if __name__ == '__main__':
    show(synthetic(2, 50))

