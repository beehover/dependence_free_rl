#!/usr/bin/python3

from scipy import stats
import numpy


def get_samples(filename):
    samples = list()

    with open(filename) as f:
        for line in f:
            fields = line.strip().split()
            samples.append(float(fields[-1]))
    return samples


deep_samples = get_samples("deep2.log")
minwaste_samples = get_samples("minwaste.log")

a = numpy.array(deep_samples)
b = numpy.array(minwaste_samples)

print("deep mean:", a.mean())
print("deep stddev:", a.std())
print("minwaste mean:", b.mean())
print("minwaste stddev:", b.std())
print(stats.ttest_ind(deep_samples, minwaste_samples, equal_var=False))
