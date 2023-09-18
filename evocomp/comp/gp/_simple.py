"""
Simple EA
created by Yifan He (heyif@outlook.com)
on Sept. 16, 2023
"""
import random
import copy
import numpy as np
from typing import Sequence
from multiprocessing import Pool
from deap import algorithms, creator, tools, gp
from deap.base import Toolbox


def simple(pset: gp.PrimitiveSet,
           toolbox: Toolbox,
           population_size: int,
           max_generation: int,
           tournament_size: int = 7,
           crossover_rate: int = 0.8,
           mutation_rate: int = 0.1,
           init_tree_height: Sequence[int] = (3, 6),
           mut_tree_height: Sequence[int] = (0, 5),
           max_tree_height: int = 10,
           parallelism: int = None,
           seed: int = None,
           verbose: bool = False):
    if seed:
        random.seed(seed)

    toolbox_ = copy.copy(toolbox)

    toolbox_.register("expr",
                      gp.genHalfAndHalf,
                      pset=pset,
                      min_=init_tree_height[0],
                      max_=init_tree_height[1])
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
    toolbox_.register("select",
                      tools.selTournament,
                      tournsize=tournament_size)
    toolbox_.register("mate",
                      gp.cxOnePoint)
    toolbox_.register("expr_mut",
                      gp.genFull,
                      min_=mut_tree_height[0],
                      max_=mut_tree_height[1])
    toolbox_.register("mutate",
                      gp.mutUniform,
                      expr=toolbox_.expr_mut,
                      pset=pset)

    toolbox_.decorate("mate",
                      gp.staticLimit(key=height,
                                    max_value=max_tree_height))
    toolbox_.decorate("mutate",
                      gp.staticLimit(key=height,
                                    max_value=max_tree_height))

    pop = toolbox_.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", lambda x: round(np.mean(x),3))
    mstats.register("std", lambda x: round(np.std(x),3))
    mstats.register("min", lambda x: round(np.min(x),3))
    mstats.register("max", lambda x: round(np.max(x),3))

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (mstats.fields if mstats else [])

    algorithms.eaSimple(pop,
                        toolbox_,
                        crossover_rate,
                        mutation_rate,
                        max_generation-1,
                        mstats,
                        hof,
                        verbose)

    solution = hof[0]
    return solution


def height(t):
    return t.height
