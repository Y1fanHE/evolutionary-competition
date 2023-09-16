"""
Simple EA
created by Yifan He (heyif@outlook.com)
on Sept. 16, 2023
"""
import operator, random
import numpy as np
from typing import Sequence
from multiprocessing import Pool
from deap import creator, tools, gp
from deap.base import Toolbox
from deap.algorithms import varAnd


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

    if parallelism:
        pool = Pool(parallelism)

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

    # TODO: multiprocessing not working with these lines below!
    # toolbox.decorate("mate",
    #                  gp.staticLimit(key=lambda t: t.height,
    #                                 max_value=max_tree_height))
    # toolbox.decorate("mutate",
    #                  gp.staticLimit(key=lambda t: t.height,
    #                                 max_value=max_tree_height))

    pop = toolbox.population(n=population_size)
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

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    if not parallelism:
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    else:
        fitnesses = pool.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if hof is not None:
        hof.update(pop)

    record = mstats.compile(pop) if mstats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, max_generation):
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in pop]

        for i in range(1, len(offspring), 2):
            if random.random() < crossover_rate:
                child1, child2 = toolbox.mate(offspring[i - 1],
                                              offspring[i])
                if child1.height <= max_tree_height and \
                    child2.height <= max_tree_height:
                    offspring[i - 1], offspring[i] = child1, child2
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                child, = toolbox.mutate(offspring[i])
                if child.height <= max_tree_height:
                    offspring[i] = child
                del offspring[i].fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if not parallelism:
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        else:
            fitnesses = pool.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if hof is not None:
            hof.update(offspring)

        pop[:] = offspring

        record = mstats.compile(pop) if mstats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    solution = hof[0]
    return solution
