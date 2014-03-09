import random
import math
import inspect
from matplotlib import pyplot

def points(n):
    return [(random.random(), random.random()) for x in range(n)]

def split_by_classifier(classifier, points):
    return [p for p in points if classifier(p)], [p for p in points if not classifier(p)]

def accuracy(classifier, truth, points):
    return sum(1 for p in points if classifier(p) == truth(p))/float(len(points))

def test_classifier(classifier, truth, n):
    return accuracy(classifier, truth, points(n))

def entropy(truth, points):
    g1, g2 = split_by_classifier(truth, points)
    if len(g1) == 0 or len(g2) == 0: return float('nan')
    return sum(-len(g)/float(len(points))*math.log(len(g)/float(len(points))) for g in [g1, g2])

def info_gain(classifier, truth, points):
    return entropy(truth, points) - sum(len(g)/float(len(points))*entropy(truth, g) for g in split_by_classifier(classifier, points))

def safe(value, replace=0):
    return replace if math.isnan(value) else value

def display_info_gain(classifier, truth, points):
    print '----'
    print 'Info gain of'
    print classifier
    print safe(info_gain(classifier, truth, points))
    trues, falses = split_by_classifier(truth, points)
    (g1_trues, g1_falses), (g2_trues, g2_falses) = [
         split_by_classifier(truth, g)
         for g in split_by_classifier(classifier, points)]
    print len(trues),':',len(falses), 'split into', len(g1_trues),':',len(g1_falses),'and', len(g2_trues),':',len(g2_falses)

def plot_points(points, truth, classifier):
    (g1_trues, g1_falses), (g2_trues, g2_falses) = [
         split_by_classifier(truth, g)
         for g in split_by_classifier(classifier, points)]
    for dataset, style in zip([g1_trues, g2_trues, g1_falses, g2_falses], ['bo', 'ro', 'bx', 'rx']):
        if dataset:
            pyplot.plot(*(zip(*dataset) + [style,]))
    pyplot.show()

def classifiers(n):
    return [Classifier(which, i/float(n)) for which in ['x', 'y'] for i in range(1,n)]

class Classifier(object):
    def __init__(self, which, thresh):
        self.which = which
        self.thresh = thresh
        self.true_child = None
        self.false_child = None
        self._values = []
    @property
    def values(self):
        if self.true_child is None:
            assert self.false_child is None
            return self._values
    @property
    def true_values(self):
        return self.true_child.values()
    @property
    def false_values(self):
        return self.false_child.values()
    def __call__(self, (x, y)):
        return {'x':x,'y':y}[self.which] > self.thresh
    def train(self, (x, y), value):
        {True: self.true_values, False: self.false_values}[value].append(self((x, y)))
    def expected(self, branch):
        return bool(round(sum(self.true_values) /
                    float(len(self.true_values if branch else self.false_values))))
    def predict(self, (x, y)):
        return self.expected(self((x, y)))
    def accuracy(self, points, truth):
        return sum(1 for p in points if self.predict(p) == truth(p))/float(len(points))
    def __repr__(self):
        return 'Classifier: %s > %s' % (self.which, self.thresh)

if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)

    def choice((x, y)):
        return x > .5

    def x_gt_y((x, y)):
        return x > y

    def simple((x, y)):
        return x > 4*y

    pnts = points(1000)

    truth = simple

    clss = classifiers(10)
    best_cls = max(clss, key=lambda c: info_gain(c, truth, pnts))
    worst_cls = min(clss, key=lambda c: info_gain(c, truth, pnts))
    display_info_gain(best_cls, truth, pnts)
    plot_points(pnts, truth, best_cls)
    display_info_gain(worst_cls, truth, pnts)
    plot_points(pnts, truth, worst_cls)

    for p in pnts:
        best_cls.train(p, truth(p))

    print 'accuracy:', best_cls.accuracy(pnts, truth)
    print entropy(best_cls, pnts)
    #plot_points(pnts, truth, best_cls)
