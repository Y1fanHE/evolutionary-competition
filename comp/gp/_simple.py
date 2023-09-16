"""
Simple EA
created by Yifan He (heyif@outlook.com)
on Sept. 16, 2023
"""
import operator, random
import numpy as np
from typing import Sequence
from deap import creator, tools, gp
from deap.base import Toolbox
from deap.algorithms import eaSimple


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
           seed: int = None,
           verbose: bool = False):
    if seed:
        random.seed(seed)

    toolbox.register("expr",
                     gp.genHalfAndHalf,
                     pset=pset,
                     min_=init_tree_height[0],
                     max_=init_tree_height[1])
    toolbox.register("individual",
                     tools.initIterate,
                     creator.Individual,
                     toolbox.expr)
    toolbox.register("population",
                     tools.initRepeat,
                     list,
                     toolbox.individual)
    toolbox.register("select",
                     tools.selTournament,
                     tournsize=tournament_size)
    toolbox.register("mate",
                     gp.cxOnePoint)
    toolbox.register("expr_mut",
                     gp.genFull,
                     min_=mut_tree_height[0],
                     max_=mut_tree_height[1])
    toolbox.register("mutate",
                     gp.mutUniform,
                     expr=toolbox.expr_mut,
                     pset=pset)
    toolbox.decorate("mate",
                     gp.staticLimit(key=operator.attrgetter("height"),
                                    max_value=max_tree_height))
    toolbox.decorate("mutate",
                     gp.staticLimit(key=operator.attrgetter("height"),
                                    max_value=max_tree_height))

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", lambda x: round(np.mean(x),3))
    mstats.register("std", lambda x: round(np.std(x),3))
    mstats.register("min", lambda x: round(np.min(x),3))
    mstats.register("max", lambda x: round(np.max(x),3))

    pop, _ = eaSimple(pop,
                      toolbox,
                      crossover_rate,
                      mutation_rate,
                      max_generation-1,
                      stats=mstats,
                      halloffame=hof,
                      verbose=verbose)
    solution = hof[0]
    return solution
