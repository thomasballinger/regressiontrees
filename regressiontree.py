import random
import math
from collections import namedtuple
import itertools


class Classifier(object):
    def __init__(self, which, thresh):
        self.which = which
        self.thresh = thresh
    def __call__(self, indvars):
        return indvars[self.which] > self.thresh
    def __repr__(self):
        return 'Classifier(%s, %s)' % (self.which, self.thresh)
    def __str__(self):
        return 'f(in) -> in[%s] > %s' % (self.which, self.thresh)
    def display(self, observation):
        if hasattr(observation[0], '_fields') and isinstance(self.which, int):
            return '%s > %s' % (observation[0]._fields[self.which], self.thresh)
        if hasattr(observation[0], 'keys'):
            return '%s > %s' % (self.which, self.thresh)
        return 'data[%s] > %s' % (self.which, self.thresh)
    @classmethod
    def build(cls, n, indvars):
        """Returns classifiers for dividing each independent variable domain into n regions"""
        return [Classifier(which, i/float(n)) for which in indvars for i in range(1,n)]

class RegressionTree(object):
    """
    >>> t = RegressionTree([((1,2), True), ((2, 3), False)]) #TODO  make non-binary work!
    >>> t.value
    True
    >>> t.observations
    [((1, 2), True), ((2, 3), False)]
    >>> t.grow()
    False
    """
    def __init__(self, observations, binary=None):
        """Builds a tree. Data is of form ((indvar1, indvar2, indvar3, ...), depvar)
        Ideally a named tuple of input data.

        Condition is True for left children, False for right children
        """
        self._observations = observations
        self.condition = None
        self.left = None
        self.right = None
        if binary is None:
            self.binary = len(set(depvar for indvars, depvar in observations)) == 2
        else:
            self.binary = binary
        if not self.binary:
            raise NotImplemented("don't understand how to calculate information gain for continuous dependant variables")

    def __repr__(self):
        if self.uniform:
            return "RegressionTree(%s, %d observations)" % (self.value, len(self.observations))
        if self.condition is None:
            return "RegressionTree(ave %s, %d observations)" % (self.value, len(self.observations))
        return "RegressionTree(%s)" % (self.condition)

    def __str__(self):
        indent = lambda x: '\n'.join([' '*4+line for line in x.split('\n')])
        if self.left is None:
            return repr(self)
        return "RegressionTree(%d samples)\n%s" % (len(self.observations), indent(self._str_helper()))

    def _str_helper(self):
        indent = lambda x: '\n'.join([' '*4+line for line in x.split('\n')])
        if self.uniform:
            return "%s (%d observations)" % (self.value, len(self.observations))
        if self.condition is None:
            return "Unexpanded: ave %s, %d observations: %r)" % (self.value, len(self.observations), self.observations)
        return "if %s\n%s\nelse\n%s" % (self.condition.display(self.observations[0]),
                                                              indent(self.left._str_helper()),
                                                              indent(self.right._str_helper()))

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
            mean = sum(depvar for indvars, depvar in self.observations) / float(len(self.observations))
            return bool(round(mean)) if self.binary else mean

    @property
    def uniform(self):
        return len(set(depvar for indvars, depvar in self.observations)) == 1

    def grow(self):
        if self.left:
            assert self.right and self.condition
            return self.left.grow() or self.right.grow()
        if not self._observations:
            return False
        classifiers = Classifier.build(10, range(len(self.observations[0][0])))
        if self.binary:
            best = max(classifiers, key=lambda c: safe(discrete_info_gain(c, self.observations)))
            left_group, right_group = split_by_classifier(best, self.observations)
            if not all([left_group, right_group]):
                return False
            self.condition = best
            self.left = RegressionTree(left_group, binary=self.binary)
            self.right = RegressionTree(right_group, binary=self.binary)
        else:
            raise NotImplemented("how does entropy / info gain work?")

def split_by_classifier(classifier, obs):
    """
    >>>

    """
    trues = []
    falses = []
    for indvars, depvar in obs:
        (trues if classifier(indvars) else falses).append((indvars, depvar))
    return trues, falses

def discrete_entropy(obs):
    """
    >>> round(discrete_entropy([[0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0]]), 4)
    0.9852
    """
    key = lambda ob: ob[1]
    groups = [list(x[1]) for x in itertools.groupby(sorted(obs, key=key), key=key)]
    if len(groups) < 2: return 0
    return sum(-len(g)/float(len(obs))*math.log(len(g)/float(len(obs)), 2) for g in groups)

def discrete_info_gain(classifier, obs):
    """
    >>> round(discrete_info_gain(lambda x:x[0]==1, [[[1],3],[[2],2],[[2],4],[[2],3],[[2],3],[[1],3],[[1],3],[[1],3]]), 4)
    0.3113
    """
    return discrete_entropy(obs) - sum(len(g)/float(len(obs))*discrete_entropy(g) for g in split_by_classifier(classifier, obs))

def safe(value, replace=0):
    return replace if math.isnan(value) else value





#test functions
def points(n):
    point = namedtuple('point', ('x', 'y'))
    return [point(random.random(), random.random()) for x in range(n)]

def circle((x, y)):
    return (x - .5)**2 + (y - .5)**2 < .3**2




import doctest
doctest.testmod(optionflags=doctest.ELLIPSIS)

discrete_entropy([[0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0]])


training_data = [(p, circle(p)) for p in points(100)]
t = RegressionTree(training_data)
for i in range(10):
    print
    print t
    t.grow()
