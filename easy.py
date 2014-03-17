"""Really easy synthetic data

Generate "eyes," 20x20 matrices of grayscale values
Figure out where the center is in them
"""



import random
import numpy
from pylab import imshow
import matplotlib.pyplot

# Let's always use 1 to 100 as the size!

def synthetic(x, y, width=100, height=100, radius=20):
    in_circle = lambda xi, yi: ((x-xi)**2 + (y-yi)**2) < radius**2
    return numpy.random.rand(height, width) + numpy.array([[1 if in_circle(i, j) else 0
                                                            for i in range(width)]
                                                           for j in range(height)])

class Feature(object):
    @classmethod
    def random(cls):
        r = lambda: int(random.random() * 100)
        return cls((r(), r()), (r(), r()))
    def __init__(self, (x1, y1), (x2, y2)):
        self.p1 = (x1, y1)
        self.p2 = (x2, y2)
    def __call__(self, im):
        return im[self.p1] > im[self.p1]
    def __repr__(self):
        return 'Feature(%r, %r)' % (self.p1, self.p2)
    def __str___(self):
        return '(feature %r > %r)' % (self.p1, self.p2)

indent = lambda x, y: '\n'.join([y+line for line in x.split('\n')])

class DecisionTree(object):
    """
    Left is true, right is false
    """
    def __init__(self, observations):
        self._observations = observations
        self.left = None
        self.right = None
        self.feature = None

    def __repr__(self):
        if self.feature is None:
            return "RegressionTree(ave %s, %d observations)" % (self.value, len(self.observations))
        return "RegressionTree(%s)" % (self.feature)

    def __str__(self):
        if self.feature is None:
            return repr(self)
        return "RegressionTree(%d samples)\n%s" % (len(self.observations), indent(self._str_helper()), '    ')

    def _str_helper(self):
        if self.feature is None:
            return "Unexpanded: ave %s, %d observations)" % (self.value, len(self.observations))
        return "if %s\n%s\nelse\n%s" % (self.feature,
                                        indent(self.left._str_helper(),  '|   '),
                                        indent(self.right._str_helper(), '    '))

    def __call__(self, depvars):
        if self.feature is None:
            return self.value
        if self.feature(depvars):
            return self.left(depvars)
        else:
            return self.right(depvars)

    @property
    def observations(self):
        if self.left is None:
            assert self.right is None
            return self._observations
        else:
            return self.left.observations + self.right.observations

    @property
    def value(self):
        if self.observations:
            mean_x = sum(x for im, (x, y) in self.observations) / float(len(self.observations))
            mean_y = sum(y for im, (x, y) in self.observations) / float(len(self.observations))
            return (mean_x, mean_y)

    def grow(self, maxdepth):
        if maxdepth == 0:
            return False
        if self.feature:
            assert self.right and self.feature
            return self.left.grow() or self.right.grow()
        if not self._observations:
            return False
        classifiers = [Feature.random() for _ in range(10)]
        best = max(classifiers, key=lambda c: safe(discrete_info_gain(c, self.observations)))
        left_group, right_group = split_by_classifier(best, self.observations)
        if not all([left_group, right_group]):
            return False
        self.feature = best
        self.left = DecisionTree(left_group)
        self.right = DecisionTree(right_group)

def show(a2d):
    imshow(a2d)
    matplotlib.pyplot.show()

if __name__ == '__main__':
    training = [(synthetic(x, y), (x, y)) for x, y in [(int(random.random()), int(random.random())) for _ in range(100)]]
    show(training[0][0])
