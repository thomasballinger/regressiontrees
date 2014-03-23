"""Really easy synthetic data

Generate "eyes," 20x20 matrices of grayscale values
Figure out where the center is in them
"""


import math
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
        return im[self.p1] > im[self.p2]
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
            return "RegressionTree(ave (%d, %d), %d observations)" % (self.value[0], self.value[1], len(self.observations))
        return "RegressionTree(%s)" % (self.feature)

    def __str__(self):
        if self.feature is None:
            return repr(self)
        return "RegressionTree(%d samples)\n%s" % (len(self.observations), indent(self._str_helper(), '    '))

    def _str_helper(self):
        if self.feature is None:
            return "Unexpanded: ave (%d, %d), %d observations)" % (self.value[0], self.value[1], len(self.observations))
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
            return self.left.grow(maxdepth - 1) or self.right.grow(maxdepth - 1)
        if not self._observations:
            return False
        classifiers = [Feature.random() for _ in range(10)]
        best = min(classifiers, key=lambda c: standard_error(*split_by_feature(c, self.observations)))
        left_group, right_group = split_by_feature(best, self.observations)
        if not all([left_group, right_group]):
            return False
        self.feature = best
        self.left = DecisionTree(left_group)
        self.right = DecisionTree(right_group)
        return True

    def grow_to_depth(self, maxdepth):
        while self.grow(maxdepth):
            pass

def split_by_feature(feature, observations):
    """
    >>> split_by_feature(lambda x: x == 'a', [('a', (1, 2)), ('b', (-1, -2))])
    ([('a', (1, 2))], [('b', (-1, -2))])
    """
    yes = []
    no = []
    for data, (x, y) in observations:
        if feature(data):
            yes.append((data, (x, y)))
        else:
            no.append((data, (x, y)))
    return yes, no

def ave(things):
    return sum(things) / float(len(things))

def ave_pos(obs):
    return [ave(z) for z in zip(*[(x,y) for im, (x, y) in obs])]

def distsqrd((x1, y1), (x2, y2)):
    return (x1-x2)**2 + (y1-y2)**2

def group_error(obs):
    ave = ave_pos(obs)
    return sum(distsqrd(ave, (x, y)) for im, (x, y) in obs)

def standard_error(obs1, obs2):
    """
    >>> ave([1,2,3])
    2.0
    >>> ave_pos([('stuff', (1, 2)), ('stuff', (-1, -2))])
    [0.0, 0.0]
    >>> distsqrd((0,0), (3, 4))
    25
    >>> group_error([('stuff', (1, 2)), ('stuff', (-1, -2))])
    10.0
    """
    return group_error(obs1) + group_error(obs2)

def show(a2d):
    imshow(a2d)
    matplotlib.pyplot.show()

def synth():
    x, y = int(random.random()*100), int(random.random()*100)
    return (synthetic(x, y), (x, y))

def evaluate(tree, n):
    errors = []
    for _ in range(n):
        s = synth()
        predicted = tree(s[0])
        error = math.sqrt(distsqrd(s[1], predicted))
        print "real:", s[1], "predicted:", predicted, "err:", error
        errors.append(error)
    return ave(errors)

def trained_tree(n_obs, maxdepth):
    training = [synth() for _ in range(n_obs)]
    #show(training[0][0])
    d = DecisionTree(training)
    d.grow_to_depth(maxdepth)
    return d

def random_classifier(stuff):
    return (random.random() * 100, random.random() * 100)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    print 'baseline:', evaluate(random_classifier, 10)
    d = trained_tree(2000, 6)
    print d
    print "error:", evaluate(d, 10)



