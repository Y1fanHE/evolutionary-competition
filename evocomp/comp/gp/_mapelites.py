"""
MAP-Elites Algorithm
created by Yifan He (heyif@outlook.com)
on Oct. 5, 2023
"""
import random
import copy
import numpy as np
from typing import Sequence, Callable
from multiprocessing import Pool
from deap import algorithms, creator, tools, gp
from deap.base import Toolbox


def map_elites(pset: gp.PrimitiveSet,
               toolbox: Toolbox,
               population_size: int,
               max_generation: int,
               phenotypic_descriptor: Callable,
               max_random_fill: int,
               crossover_rate: int = 0.8,
               mutation_rate: int = 0.1,
               init_tree_height: Sequence[int] = (3, 6),
               mut_tree_height: Sequence[int] = (0, 5),
               max_tree_height: int = 10,
               parallelism: int = None,
               objective_sensitive: bool = False,
               return_archive: bool = False,
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
                      tools.selRandom)
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
    if not objective_sensitive:
        print("NOT objective sensitive")
        toolbox_.register("describe",
                          phenotypic_descriptor,
                          pset=pset)
    else:
        toolbox_.register("describe",
                          sensitive_phenotypic_descriptor,
                          phenotypic_descriptor=phenotypic_descriptor,
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

    archive, archive_index, _ = eaMapElites(pop,
                                            toolbox_,
                                            max_random_fill,
                                            crossover_rate,
                                            mutation_rate,
                                            max_generation-1,
                                            mstats,
                                            hof,
                                            verbose)

    if return_archive:
        return archive, archive_index

    solution = hof[0]
    return solution


def eaMapElites(population, toolbox, nfill, cxpb, mutpb, ngen, stats=None,
                halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    archive_index = {}
    archive = []

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    phenotypes = toolbox.map(toolbox.describe, invalid_ind)
    for ind, phn in zip(invalid_ind, phenotypes):
        index = archive_index.get(phn)
        if index is None:
            archive.append(ind)
            archive_index.update({phn:len(archive)-1})
        else:
            ind_ = archive[index]
            archive[index] = tools.selBest([ind, ind_], k=1)[0]

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        if len(archive) < nfill:
            offspring = toolbox.population(len(population))
        else:
            # Select the next generation individuals
            offspring = toolbox.select(archive, len(population))

            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        phenotypes = toolbox.map(toolbox.describe, invalid_ind)
        for ind, phn in zip(invalid_ind, phenotypes):
            index = archive_index.get(phn)
            if index is None:
                archive.append(ind)
                archive_index.update({phn:len(archive)-1})
            else:
                ind_ = archive[index]
                archive[index] = tools.selBest([ind, ind_], k=1)[0]

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return archive, archive_index, logbook


def height(t):
    return t.height

def sensitive_phenotypic_descriptor(individual, phenotypic_descriptor, pset):
    is_obj_eq = individual.fitness.values[-1]
    phn = phenotypic_descriptor(individual, pset)
    phn_ = [i for i in phn] + [is_obj_eq]
    return tuple(phn_)
