"""
Random Search
created by Yifan He (heyif@outlook.com)
on Oct. 23, 2023
"""
import random
import copy
from multiprocessing import Pool
from deap import creator, tools, gp
from deap.base import Toolbox


def simple(pset: gp.PrimitiveSet,
           toolbox: Toolbox,
           max_evaluation: int,
           min_tree_height: int = 3,
           max_tree_height: int = 10,
           parallelism: int = None,
           seed: int = None):
    if seed:
        random.seed(seed)

    toolbox_ = copy.copy(toolbox)

    toolbox_.register("expr",
                      gp.genHalfAndHalf,
                      pset=pset,
                      min_=min_tree_height,
                      max_=max_tree_height)
    toolbox_.register("individual",
                      tools.initIterate,
                      creator.Individual,
                      toolbox_.expr)
    toolbox_.register("population",
                      tools.initRepeat,
                      list,
                      toolbox_.individual)
    if parallelism:
        pool = Pool(parallelism)
        toolbox_.register("map", pool.map)

    pop = toolbox_.population(n=max_evaluation)
    hof = tools.HallOfFame(1)
    hof.update(pop)

    solution = hof[0]
    return solution
