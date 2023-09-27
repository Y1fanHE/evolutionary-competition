"""
Distance Metrics
created by Yifan He (heyif@outlook.com)
on Sept. 12, 2023
"""
import numpy as np
from scipy.stats import wasserstein_distance as wass_dist


def euclidean_distance(solution1, solution2):
    return np.linalg.norm(solution1-solution2)


def wasserstein_distance(population1, population2):
    pop1 = np.array(population1)
    pop2 = np.array(population2)

    dist = 0
    for i in range(len(pop1[0])):
        dist += wass_dist(pop1[:,i], pop2[:,i])
    dist /= len(pop1[0])
    return dist
